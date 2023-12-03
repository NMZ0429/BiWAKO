import os
import re
import time
from dataclasses import dataclass

import requests
from github import Github


@dataclass
class Issue:
    """A GitHub issue."""

    number: int
    title: str
    url: str
    labels: list[str]
    closed: bool
    project_items: list[dict[str, str]]
    status: str = ""
    level: str = ""
    incident_id: str = ""
    title_stripped: str = ""

    def __post_init__(self):
        """Post init function."""
        self.status = self.get_status()
        self.level = self.get_level()
        self.incident_id = self.get_incident_id()
        self.title_stripped = self.strip_title()

    def get_status(self) -> str:
        for field in self.project_items:
            if "name" in field:
                return field["name"]
        return "unknown"

    def get_level(self) -> str:
        for label in self.labels:
            if label.startswith("severity:"):
                return label.split(":")[1]
        if "enhancement" in self.labels:
            return "enhancement"

        return "unknown"

    def get_incident_id(self) -> str:
        pattern = r"\[i-(\d+)\]"
        match = re.search(pattern, self.title)

        if match:
            return f"i-{match.group(1)}"
        else:
            return ""

    def strip_title(self) -> str:
        pattern = r"\[i-(\d+)\]"
        return re.sub(pattern, "", self.title).strip()


class PrettyPrinter:
    """Pretty print the list of issues as a markdown table."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues

    def digest(self, add_diagram: bool = False) -> str:
        """Return the pretty printed string."""
        content = (
            "| Incident ID | Status | Summary | Severity | Issue |\n"
            + "| --- | --- | --- | --- | --- |\n"
            + "\n".join([self._format_issue(issue) for issue in self.issues])
        )
        if add_diagram:
            content += "\n\n"
            content += f"""
```mermaid
pie
    title Incident Progress
    "Completed" : {len(list(filter(lambda x: x.status == "Done", self.issues)))}
    "In Progress" : {len(list(filter(lambda x: x.status == "In Progress", self.issues)))}
    "To Do" : {len(list(filter(lambda x: x.status == "Todo", self.issues)))}
```
"""
        return content

    def _format_issue(self, issue: Issue) -> str:
        """Return the markdown formatted string for the issue."""
        return f"| {issue.incident_id} | {issue.status} | {issue.title_stripped} | {issue.level} | [#{issue.number}]({issue.url}) |"


def run_query(query: str, token: str):
    headers = {"Authorization": f"token {token}"}
    request = requests.post(
        "https://api.github.com/graphql", json={"query": query}, headers=headers
    )
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(
            f"Query failed to run by returning code of {request.status_code}. {query}"
        )


# def count_issues_with_label_and_project(
#     repo_owner: str,
#     repo_name: str,
#     label: str,
#     project_name: str,
#     column_names: list[str],
# ) -> tuple[int, int, list[Issue]]:
#     """Return the tuple of integers in the format (count, total) where count is the number of issues with the label and project in the columns and total is the total number of issues with the label.

#     Args:
#         repo_owner (str): The owner of the repository.
#         repo_name (str): The name of the repository.
#         label (str): The label to search for.
#         project_name (str): The name of the project.
#         column_names (list[str]): The names of the columns to search in.

#     Returns:
#         tuple[int, int]: The number of issues with the label in the project under the columns and the total number of issues with the label.
#     """
#     query = f"""
#     {{
#       repository(owner: "{repo_owner}", name: "{repo_name}") {{
#         issues(labels: "{label}", first: 100) {{
#           nodes {{
#             projectCards(first: 100) {{
#               nodes {{
#                 project {{
#                   name
#                 }}
#                 column {{
#                   name
#                 }}
#               }}
#             }}
#           }}
#         }}
#       }}
#     }}
#     """
#     result = run_query(query)
#     count = 0
#     issues = []
#     for issue in result["data"]["repository"]["issues"]["nodes"]:
#         for card in issue["projectCards"]["nodes"]:
#             if (
#                 card["project"]["name"] == project_name
#                 and card["column"]["name"] in column_names
#             ):
#                 count += 1
#                 issues.append(
#                     Issue(
#                         issue["number"],
#                         issue["title"],
#                         issue["url"],
#                         issue["labels"]["nodes"],
#                         closed=False,
#                         project_items=card["column"]["nodes"],
#                     )
#                 )

#     return count, len(result["data"]["repository"]["issues"]["nodes"]), issues


def comment_on_issue(repo_owner, repo_name, issue_number, message):
    g = Github(os.getenv("ISSUE_TRACKER_TOKEN"))
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    issue = repo.get_issue(number=issue_number)
    issue.create_comment(message)


def get_issues_in_project(
    project_number: int,
    user: str | None = "",
    organization: str | None = "",
    token: str = "",
    max_iterations=20,
) -> list[Issue]:
    """Get the issues in the project. If the user is provided, the user's projects are queried.
    If the organization is provided, the organization's projects are queried.

    Args:
        project_number (int): The project number.
        token (str): The GitHub personal access token or GitHub App token.
        user (str | None): User name owning the project.
        organization (str | None): Organization name owning the project.
        max_iterations (int, optional): Max number of iterations to query. Defaults to 20.

    Raises:
        ValueError: _description_

    Returns:
        list[Issue]: _description_
    """
    if not bool(user) != bool(organization):
        raise ValueError("Either user or organization must be provided.")
    auth = (
        f'organization(login: "{organization}")'
        if organization
        else f'user(login: "{user}")'
    )
    if not token:
        token = str(os.getenv("ISSUE_TRACKER_TOKEN"))
    cursor = "null"
    has_next_page = True
    num_request = 0
    issues = []
    while has_next_page and num_request < max_iterations:
        query = f"""
        query {{
            {auth} {{
                projectV2(number: {project_number}) {{
                    title
                    url
                    items(first: 100, after: "{cursor}") {{
                        nodes {{
                            type
                            content {{
                                ... on Issue {{
                                    assignees(first: 10) {{
                                        edges {{
                                            node {{
                                                name
                                            }}
                                        }}
                                    }}
                                    url
                                    title
                                    closed
                                    number
                                    labels(first: 10) {{
                                        nodes {{
                                            name
                                        }}
                                    }}
                                    projectItems(first: 10) {{
                                        nodes {{
                                            fieldValues(first: 10) {{
                                                nodes {{
                                                    ... on ProjectV2ItemFieldSingleSelectValue {{
                                                        name
                                                    }}
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                                ... on PullRequest {{
                                    title
                                    baseRefName
                                    closed
                                    headRefName
                                    url
                                }}
                            }}
                        }}
                        pageInfo {{
                            endCursor
                            hasNextPage
                        }}
                    }}
                }}
            }}
        }}
        """
        result = run_query(query, token)["data"][
            "organization" if organization else "user"
        ]["projectV2"]
        cursor = result["items"]["pageInfo"]["endCursor"]
        has_next_page = result["items"]["pageInfo"]["hasNextPage"]
        issues += [
            issue for issue in result["items"]["nodes"] if issue["type"] == "ISSUE"
        ]
        num_request += 1
        time.sleep(0.5)

    return [
        Issue(
            number=issue["content"]["number"],
            title=issue["content"]["title"],
            url=issue["content"]["url"],
            labels=[label["name"] for label in issue["content"]["labels"]["nodes"]],
            closed=issue["content"]["closed"],
            project_items=issue["content"]["projectItems"]["nodes"][0]["fieldValues"][
                "nodes"
            ],
        )
        for issue in issues
    ]


def update_issue_tracker(
    tracker_number: int, issues: list[Issue], token: str, owner: str, repo: str
):
    """Update the issue tracker with the list of issues.

    Args:
        tracker_number (int): The issue number which is the issue tracker.
        issues (list[Issue]): The list of issues to update.
        token (str): The GitHub personal access token.
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
    """
    g = Github(token)
    repository = g.get_repo(f"{owner}/{repo}")
    tracker = repository.get_issue(tracker_number)

    body = (
        "## Current Incident Status as of "
        + time.strftime("%Y-%m-%d %H:%M:%S")
        + "\n\n"
    )
    body += PrettyPrinter(issues).digest(add_diagram=True)

    tracker.edit(body=body)


if __name__ == "__main__":
    repo_owner = "NMZ0429"
    repo_name = "BiWAKO"
    label = "incident"
    project_number = 1
    issue_tracker = 37
    token = os.getenv("ISSUE_TRACKER_TOKEN")
    assert token, "ISSUE_TRACKER_TOKEN is not set."

    issues = get_issues_in_project(
        user=repo_owner, project_number=project_number, max_iterations=20
    )

    update_issue_tracker(
        tracker_number=issue_tracker,
        issues=issues,
        token=token,
        owner=repo_owner,
        repo=repo_name,
    )
