from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import json
from datetime import datetime
import subprocess
from pathlib import Path
import re
from typing import Dict, Any, List, Optional
import requests

# Initialize FastAPI app
app = FastAPI()

class FileSystemGuard:
    def __init__(self):
        self.base_path = "/data"
        
    def validate_path(self, path: str) -> bool:
        """Ensure path is within /data directory"""
        abs_path = os.path.abspath(os.path.join(self.base_path, path))
        return abs_path.startswith(self.base_path)

    def read_file(self, path: str) -> str:
        if not self.validate_path(path):
            raise ValueError(f"Access denied: {path}")
        with open(os.path.join(self.base_path, path), 'r') as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        if not self.validate_path(path):
            raise ValueError(f"Access denied: {path}")
        with open(os.path.join(self.base_path, path), 'w') as f:
            f.write(content)

class TaskExecutor:
    def __init__(self):
        self.file_system = FileSystemGuard()

    def execute_task(self, task_description: str) -> Dict[str, Any]:
        try:
            # Parse task description
            parsed_task = self._parse_task(task_description)
            
            result = {
                'task_type': parsed_task['type'],
                'status': 'success',
                'output': None,
                'verification_data': {}
            }

            if parsed_task['type'] == 'dates':
                result['output'] = self.count_wednesdays(parsed_task['input'])
            elif parsed_task['type'] == 'markdown':
                self.format_with_prettier(parsed_task['input'])
                result['output'] = f"Formatted {parsed_task['input']} with Prettier"
            elif parsed_task['type'] == 'contacts':
                self.sort_contacts(parsed_task['input'])
                result['output'] = "Sorted contacts and saved to /data/contacts-sorted.json"
            elif parsed_task['type'] == 'email':
                result['output'] = self._extract_email(parsed_task['input'])
            elif parsed_task['type'] == 'credit_card':
                result['output'] = self._extract_credit_card(parsed_task['input'])
            elif parsed_task['type'] == 'comments':
                result['output'] = self._find_similar_comments(parsed_task['input'])
            elif parsed_task['type'] == 'database':
                result['output'] = self._analyze_database(parsed_task['input'])
                
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'verification_data': {}
            }

    def count_wednesdays(self, file_path: str) -> int:
        """Count the number of Wednesdays in a given file."""
        return count_wednesdays(file_path)

    def format_with_prettier(self, file_path: str, prettier_version: str = "3.4.2") -> None:
        """Format the file using Prettier."""
        format_with_prettier(file_path, prettier_version)

    def sort_contacts(self, file_path: str) -> None:
        """Sort contacts based on the last_name and first_name."""
        sort_contacts(file_path)

    def _parse_task(self, task_description: str) -> Dict[str, Any]:
        """Parse task description and return a structured object."""
        # You should implement this method to parse the task description and extract the necessary details
        return {
            'type': 'dates',  # Example, update with real parsing logic
            'input': '/data/dates.txt'  # Example, change according to your task parsing logic
        }

# FastAPI route for /run task
@app.post("/run")
async def run_task(task_description: str = Query(..., alias="task")):
    if not task_description:
        raise HTTPException(status_code=400, detail="Task description is required")
        
    executor = TaskExecutor()
    result = executor.execute_task(task_description)
    
    if result['status'] == 'success':
        return result
    else:
        raise HTTPException(status_code=500, detail=str(result.get('message', 'Unknown error')))

# Additional helper functions for the tasks

def call_llm(prompt: str) -> str:
    """Call the AI Proxy to interact with GPT-4o-Mini."""
    headers = {
        "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "model": "gpt-4o-mini"
    }
    response = requests.post("https://api.aiproxy.io/v1/generate", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['output']
    else:
        raise ValueError(f"LLM request failed with status code {response.status_code}")
    
def count_wednesdays(file_path: str) -> int:
    """Count how many Wednesdays are in the dates list in the given file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    wednesday_count = 0
    for line in lines:
        date_str = line.strip()
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            if date_obj.weekday() == 2:  # Wednesday
                wednesday_count += 1
        except ValueError:
            continue  # Skip invalid dates
    return wednesday_count

import subprocess

def format_with_prettier(file_path: str, prettier_version: str = "3.4.2") -> None:
    """Use Prettier to format a markdown file."""
    result = subprocess.run(
        ['npx', f'prettier@{prettier_version}', '--write', file_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Prettier failed: {result.stderr}")


import json

def sort_contacts(file_path: str) -> None:
    """Sort contacts by last_name, then first_name."""
    with open(file_path, 'r') as f:
        contacts = json.load(f)
    
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
    
    with open('/data/contacts-sorted.json', 'w') as f:
        json.dump(sorted_contacts, f, indent=2)

# Run the app with Uvicorn (this will run the FastAPI app when using `uvicorn`):
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
