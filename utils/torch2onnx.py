import argparse

import numpy as np
import onnx
import timm
import torch


parser = argparse.ArgumentParser(description="PyTorch Image Model Conversion")
parser.add_argument("output", metavar="ONNX_FILE", help="output model filename")
parser.add_argument(
    "--model",
    "-m",
    metavar="MODEL",
    default="vit_base_resnet50d_224",
    help="model architecture (default: vit_base_resnet50d_224)",
)
parser.add_argument(
    "--opset", type=int, default=10, help="ONNX opset to use (default: 10)"
)
parser.add_argument(
    "--simplify",
    "-s",
    action="store_true",
    default=False,
    help="use onnxsim to simplify the model",
)
parser.add_argument(
    "--dynamic-size",
    action="store_true",
    default=False,
    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.',
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1)",
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--num-classes", type=int, default=1000, help="Number classes in dataset"
)
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to checkpoint (default: none)",
)
parser.add_argument(
    "--keep-init",
    action="store_true",
    default=False,
    help="Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.",
)
parser.add_argument(
    "--aten-fallback",
    action="store_true",
    default=False,
    help="Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.",
)


def simplify_model(
    onnx_model: onnx.ModelProto, input_tensor: torch.Tensor, destination: str
) -> None:
    """Save a simplified version of the model to the destination path

    Args:
        onnx_model (onnx.ModelProto): The model to simplify
        input_tensor (torch.Tensor): The input tensor to the model. Must have the static shape
        destination (str): The path to save the simplified model to
    """
    print("==> Simplifying model")
    import onnxsim

    dest = destination.replace(".onnx", "") + "_simplified.onnx"
    b, c, h, w = input_tensor.shape

    static_shape = {"input0": [b, c, h, w]}
    simple_model, is_valid = onnxsim.simplify(
        onnx_model, input_shapes=static_shape  # type:ignore
    )
    assert is_valid, "Simplifier failed to simplify the model"
    onnx.save_model(simple_model, dest)
    print(f"==> Simplified model saved to {dest}")


def check_caffe2(
    onnx_model: onnx.ModelProto, output_tensor: torch.Tensor, destination: str
) -> None:
    """Validate the caffe2 model

    Args:
        onnx_model (onnx.ModelProto): The model to check the compatibility with caffe2
        output_tensor (torch.Tensor): The output tensor of the model at conversion
        destination (str): The path of the model to check the compatibility with caffe2
    """
    import caffe2.python.onnx.backend as onnx_caffe2

    print(
        f"==> Loading model into Caffe2 backend and comparing forward pass. {destination}"
    )
    caffe2_backend = onnx_caffe2.prepare(onnx_model)
    B = {onnx_model.graph.input[0].name: x.data.numpy()}  # type: ignore
    c2_out = caffe2_backend.run(B)[0]
    np.testing.assert_almost_equal(output_tensor.data.numpy(), c2_out, decimal=5)
    print("==> Passed")


def main():
    args = parser.parse_args()

    args.pretrained = False if args.checkpoint else True

    print("==> Creating PyTorch {} model".format(args.model))
    print(f"ONNX version {onnx.__version__} and PyTorch version {torch.__version__}")

    model = timm.create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        exportable=True,
    )

    model.eval()

    example_input = torch.randn(
        (args.batch_size, 3, args.img_size or 224, args.img_size or 224),
        requires_grad=True,
    )
    try:
        model(example_input)
    except:
        raise RuntimeError(
            f"Model failed to run on example input of shape {example_input.shape}"
        )

    print("==> Exporting model to ONNX format at '{}'".format(args.output))
    input_names = ["input0"]
    output_names = ["output0"]
    dynamic_axes = {"input0": {0: "batch"}, "output0": {0: "batch"}}
    if args.dynamic_size:
        dynamic_axes["input0"][2] = "height"
        dynamic_axes["input0"][3] = "width"
    if args.aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    print(
        "==> If you see the RuntimeError for unsupported operator, check https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid to see if the operator is supported by onnx."
    )

    torch_out = torch.onnx._export(
        model,
        example_input,
        args.output,
        export_params=True,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=args.keep_init,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        operator_export_type=export_type,
    )

    print("==> Loading and checking exported model from '{}'".format(args.output))
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)  # type:ignore # assuming throw on error
    print("==> Passed")

    if args.simplify:
        simplify_model(
            onnx_model=onnx_model, input_tensor=example_input, destination=args.output
        )
    if args.keep_init and args.aten_fallback:
        check_caffe2(
            onnx_model=onnx_model, output_tensor=torch_out, destination=args.output
        )


if __name__ == "__main__":
    main()
