from typing import Any, Dict
from pathlib import Path
from urllib.parse import urlparse
import asyncio
import os
import subprocess
from ...utils.debug_log import debug_log
from ..package_resolver import get_local_package_path

def index_repository(graph_builder, job_manager, loop, list_repos_func, **args) -> Dict[str, Any]:
    """
    Clone a git repository by HTTPS URL and index it into the graph.
    Injects GIT_TOKEN into the URL for authenticated cloning if available.
    """
    url = args.get("url")
    branch = args.get("branch", "main")
    is_dependency = args.get("is_dependency", False)

    if not url:
        return {"error": "The 'url' parameter is required."}

    try:
        # Derive repo name from URL (last path segment, strip .git)
        parsed = urlparse(url)
        repo_name = parsed.path.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        if not repo_name:
            return {"error": f"Could not derive repository name from URL: {url}"}

        clone_path = Path("/workspace") / repo_name
        path_obj = clone_path.resolve()

        # Check if already indexed
        indexed_repos = list_repos_func().get("repositories", [])
        for repo in indexed_repos:
            if Path(repo["path"]).resolve() == path_obj:
                return {
                    "success": False,
                    "message": f"Repository '{repo_name}' is already indexed at {path_obj}."
                }

        # Build clone URL with token if GIT_TOKEN is available
        clone_url = url
        git_token = os.environ.get("GIT_TOKEN")
        if git_token and parsed.hostname:
            clone_url = f"{parsed.scheme}://{git_token}@{parsed.hostname}"
            if parsed.port:
                clone_url += f":{parsed.port}"
            clone_url += parsed.path

        # Clone the repository
        debug_log(f"Cloning {url} (branch: {branch}) into {clone_path}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, clone_url, str(clone_path)],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            # Sanitize error output to avoid leaking tokens
            stderr = result.stderr
            if git_token:
                stderr = stderr.replace(git_token, "***")
            return {"error": f"git clone failed: {stderr.strip()}"}

        # Index the cloned repository
        total_files, estimated_time = graph_builder.estimate_processing_time(path_obj)
        job_id = job_manager.create_job(str(path_obj), is_dependency)
        job_manager.update_job(job_id, total_files=total_files, estimated_duration=estimated_time)

        coro = graph_builder.build_graph_from_path_async(
            path_obj, is_dependency, job_id
        )
        asyncio.run_coroutine_threadsafe(coro, loop)

        debug_log(f"Started background job {job_id} for cloned repo: {repo_name} at {path_obj}")

        return {
            "success": True, "job_id": job_id,
            "repository": repo_name,
            "clone_path": str(path_obj),
            "branch": branch,
            "message": f"Cloned '{repo_name}' and started indexing",
            "estimated_files": total_files,
            "estimated_duration_seconds": round(estimated_time, 2),
            "estimated_duration_human": f"{int(estimated_time // 60)}m {int(estimated_time % 60)}s" if estimated_time >= 60 else f"{int(estimated_time)}s",
            "instructions": f"Use 'check_job_status' with job_id '{job_id}' to monitor progress"
        }

    except subprocess.TimeoutExpired:
        return {"error": f"git clone timed out after 300 seconds for {url}"}
    except Exception as e:
        debug_log(f"Error in index_repository: {str(e)}")
        return {"error": f"Failed to clone and index repository: {str(e)}"}


def add_code_to_graph(graph_builder, job_manager, loop, list_repos_func, **args) -> Dict[str, Any]:
    """
    Tool implementation to index a directory of code.
    Runs indexing asynchronously via a background job.
    """
    path = args.get("path")
    is_dependency = args.get("is_dependency", False)
    
    try:
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            return {
                "success": True,
                "status": "path_not_found",
                "message": f"Path '{path}' does not exist."
            }

        # Prevent re-indexing the same repository.
        indexed_repos = list_repos_func().get("repositories", [])
        for repo in indexed_repos:
            if Path(repo["path"]).resolve() == path_obj:
                return {
                    "success": False,
                    "message": f"Repository '{path}' is already indexed."
                }
        
        # Estimate time and create a job for the user to track.
        total_files, estimated_time = graph_builder.estimate_processing_time(path_obj)
        job_id = job_manager.create_job(str(path_obj), is_dependency)
        job_manager.update_job(job_id, total_files=total_files, estimated_duration=estimated_time)
        
        # Create the coroutine for the background task and schedule it on the main event loop.
        coro = graph_builder.build_graph_from_path_async(
            path_obj, is_dependency, job_id
        )
        asyncio.run_coroutine_threadsafe(coro, loop)
        
        debug_log(f"Started background job {job_id} for path: {str(path_obj)}, is_dependency: {is_dependency}")
        
        return {
            "success": True, "job_id": job_id,
            "message": f"Background processing started for {str(path_obj)}",
            "estimated_files": total_files,
            "estimated_duration_seconds": round(estimated_time, 2),
            "estimated_duration_human": f"{int(estimated_time // 60)}m {int(estimated_time % 60)}s" if estimated_time >= 60 else f"{int(estimated_time)}s",
            "instructions": f"Use 'check_job_status' with job_id '{job_id}' to monitor progress"
        }
    
    except Exception as e:
        debug_log(f"Error creating background job: {str(e)}")
        return {"error": f"Failed to start background processing: {str(e)}"}

def add_package_to_graph(graph_builder, job_manager, loop, list_repos_func, **args) -> Dict[str, Any]:
    """Tool to add a package to the graph by auto-discovering its location"""
    package_name = args.get("package_name")
    language = args.get("language")
    is_dependency = args.get("is_dependency", True)

    if not language:
        return {"error": "The 'language' parameter is required."}

    try:
        # Check if the package is already indexed
        indexed_repos = list_repos_func().get("repositories", [])
        for repo in indexed_repos:
            if repo.get("is_dependency") and (repo.get("name") == package_name or repo.get("name") == f"{package_name}.py"):
                return {
                    "success": False,
                    "message": f"Package '{package_name}' is already indexed."
                }

        package_path = get_local_package_path(package_name, language)
        
        if not package_path:
            return {"error": f"Could not find package '{package_name}' for language '{language}'. Make sure it's installed."}
        
        if not os.path.exists(package_path):
            return {"error": f"Package path '{package_path}' does not exist"}
        
        path_obj = Path(package_path)
        
        total_files, estimated_time = graph_builder.estimate_processing_time(path_obj)
        
        job_id = job_manager.create_job(package_path, is_dependency)
        
        job_manager.update_job(job_id, total_files=total_files, estimated_duration=estimated_time)
        
        coro = graph_builder.build_graph_from_path_async(
            path_obj, is_dependency, job_id
        )
        asyncio.run_coroutine_threadsafe(coro, loop)
        
        debug_log(f"Started background job {job_id} for package: {package_name} at {package_path}, is_dependency: {is_dependency}")
        
        return {
            "success": True, "job_id": job_id, "package_name": package_name,
            "discovered_path": package_path,
            "message": f"Background processing started for package '{package_name}'",
            "estimated_files": total_files,
            "estimated_duration_seconds": round(estimated_time, 2),
            "estimated_duration_human": f"{int(estimated_time // 60)}m {int(estimated_time % 60)}s" if estimated_time >= 60 else f"{int(estimated_time)}s",
            "instructions": f"Use 'check_job_status' with job_id '{job_id}' to monitor progress"
        }
    
    except Exception as e:
        debug_log(f"Error creating background job for package {package_name}: {str(e)}")
        return {"error": f"Failed to start background processing for package '{package_name}': {str(e)}"}
