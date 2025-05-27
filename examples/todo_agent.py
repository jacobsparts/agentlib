#!/usr/bin/env python3
import os
import datetime
import sqlite3
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field
from agentlib import BaseAgent

class Agent(BaseAgent):
    model = 'google/gemini-2.5-flash'

class TaskProcessingAgent(Agent):
    system = "You are a task processing agent. You will receive structured task data to schedule, store in a database, and retrieve when requested. You handle creating new tasks, listing all tasks, and filtering tasks by priority or date."

    def __init__(self):
        super().__init__()
        self.db_path = Path("tasks.db")
        self._conn = None

    @property
    def conn(self):
        """Get the database connection, initializing it if needed."""
        if self._conn is None:
            self._init_db()
        return self._conn

    def _init_db(self):
        """Initialize the SQLite database and create the tasks table if it doesn't exist."""
        self._conn = sqlite3.connect(self.db_path)
        cursor = self._conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            date TEXT,
            priority TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        self._conn.commit()

    @Agent.tool
    def schedule_new_task(self,
                          description: str = "The detailed description of the task.",
                          date: Optional[datetime.date] = "Optional date for the task.",
                          priority: Literal["high", "medium", "low"] = "The priority of the task."):
        """Schedules a new task with the given details and persists it to the database."""
        self.complete = True
        
        # Use the persistent connection
        cursor = self.conn.cursor()
        
        # Insert the task into the database
        created_at = datetime.datetime.now().isoformat()
        date_str = str(date) if date else None
        
        cursor.execute(
            "INSERT INTO tasks (description, date, priority, created_at) VALUES (?, ?, ?, ?)",
            (description, date_str, priority, created_at)
        )
        
        task_id = cursor.lastrowid
        self.conn.commit()
        
        _due = f" due {date}" if date else ''
        resp = f"Task '{description}'{_due} with {priority} priority has been scheduled and saved to database with ID {task_id}."
        return resp
        
    @Agent.tool
    def get_all_tasks(self):
        """Retrieves all tasks from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, description, date, priority, created_at FROM tasks ORDER BY id DESC")
        tasks = cursor.fetchall()
        
        if not tasks:
            return "No tasks found in the database."
        
        result = "Tasks in the database:\n"
        for task in tasks:
            task_id, description, date, priority, created_at = task
            date_info = f" due {date}" if date else ""
            result += f"- ID {task_id}: '{description}'{date_info} with {priority} priority (created: {created_at})\n"
        
        return result
    
    @Agent.tool
    def get_tasks_by_priority(self, priority: Literal["high", "medium", "low"] = "The priority to filter by."):
        """Retrieves tasks with the specified priority."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, description, date, priority, created_at FROM tasks WHERE priority = ? ORDER BY id DESC", (priority,))
        tasks = cursor.fetchall()
        
        if not tasks:
            return f"No tasks with {priority} priority found in the database."
        
        result = f"Tasks with {priority} priority:\n"
        for task in tasks:
            task_id, description, date, priority, created_at = task
            date_info = f" due {date}" if date else ""
            result += f"- ID {task_id}: '{description}'{date_info} (created: {created_at})\n"
        
        return result
    
    @Agent.tool
    def get_tasks_by_date(self, date: datetime.date = "The date to filter by."):
        """Retrieves tasks scheduled for the specified date."""
        date_str = str(date)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, description, date, priority, created_at FROM tasks WHERE date = ? ORDER BY id DESC", (date_str,))
        tasks = cursor.fetchall()
        
        if not tasks:
            return f"No tasks scheduled for {date} found in the database."
        
        result = f"Tasks scheduled for {date}:\n"
        for task in tasks:
            task_id, description, date, priority, created_at = task
            result += f"- ID {task_id}: '{description}' with {priority} priority (created: {created_at})\n"
        
        return result

class TaskAgent(Agent):
    system = (
        "You are a friendly and efficient task scheduling assistant. "
        "Your goal is to understand what task the user wants to add, including its description, "
        "an optional due date, and an optional priority (high, medium, or low; defaults to medium if unspecified). "
        "Use the 'get_current_date' tool if the user mentions relative dates like 'tomorrow' or 'next Friday'.  "
        "Once you have gathered all necessary details (description is mandatory), "
        "call the 'process_task_request' tool to finalize and delegate the task. "
        "You must fulfill the user's request without further input. "
        "Answer any followup questions that the user may have. "
        "You can retrieve tasks using 'get_all_tasks', 'get_tasks_by_priority', or 'get_tasks_by_date' tools. "
        "If the user asks for a summary or to see their tasks, use the appropriate retrieval tool. "
        "You may summarize from the current chat transcript as needed. "
    )

    def __init__(self):
        super().__init__()
        self.task_processing_agent = TaskProcessingAgent()

    @Agent.tool
    def get_current_date(self):
        """Returns the current date in YYYY-MM-DD format. Use this to resolve relative dates like 'today' or 'tomorrow' before calling process_task_request."""
        return datetime.datetime.now().strftime("%A, %B %d, %Y")

    @Agent.tool
    def process_task_request(self,
                             description: str = "The detailed description of the task.",
                             date: Optional[datetime.date] = "Optional date for the task.",
                             priority: Literal["high", "medium", "low"] = "The priority of the task ('high', 'medium', 'low'). Defaults to 'medium'."):
        """Processes the captured task details by delegating to the TaskProcessingAgent."""
        
        task_details_parts = [f"description: '{description}'"]
        if date:
            task_details_parts.append(f"date: '{date}'")
        task_details_parts.append(f"priority: '{priority}'")
        
        task_message_for_processing_agent = f"Please schedule the task with the following details: {', '.join(task_details_parts)}."
        
        # We're using the TaskProcessingAgent instance that was created in __init__
        # This allows us to maintain state between calls
        return self.task_processing_agent.run(task_message_for_processing_agent)
        
    @Agent.tool
    def get_all_tasks(self):
        """Retrieves all tasks from the database."""
        return self.task_processing_agent.get_all_tasks()
    
    @Agent.tool
    def get_tasks_by_priority(self, priority: Literal["high", "medium", "low"] = "The priority to filter by."):
        """Retrieves tasks with the specified priority."""
        return self.task_processing_agent.get_tasks_by_priority(priority)
    
    @Agent.tool
    def get_tasks_by_date(self, date: datetime.date = "The date to filter by."):
        """Retrieves tasks scheduled for the specified date."""
        return self.task_processing_agent.get_tasks_by_date(date)
    
    @Agent.tool
    def complete(self, message: str = "Message to user"):
        """Reply to the user to indicate task completion."""
        self.complete = True
        return message


def main():
    if not Path("tasks.db").exists():
        agent = TaskAgent()
        print("----- Running task creation demo -----")
        for msg in [
            "I need to buy milk.",
            "Remind me to wish my brother Happy Birthday tomorrow.  And don't let me forget about dinner with Juliet on Friday.",
            "Add 'finish report' for tomorrow, high priority.",
            "Hey, can you remind me to call John? It's pretty important. Maybe sometime next week?"
        ]:
            print("User:", msg)
            print("Agent:", agent.run(msg))

        # For the summary, we can use chat since it doesn't require tool execution
        msg = "Give me a summary of today's requests. Respond directly with a plaintext list."
        print(f"User: {msg}\nAgent: {agent.chat(msg)}")
        print(f"\n----- Execute the script again to see the retrieval demo -----")
    
    else:
        agent = TaskAgent()
        print("----- Running task retrieval demo -----")
        for msg in [
            "Show me all my tasks",
            "What high priority tasks do I have?",
            "What do I have scheduled for tomorrow?",
        ]:
                print("User:", msg)
                print("Agent:", agent.run(msg))

if __name__ == "__main__":
    main()
