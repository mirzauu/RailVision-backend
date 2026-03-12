"""
Todo Management Tool for Supervisor Agent State Management

This tool allows the supervisor agent to create, update, and track todo items
for long-running tasks, providing state management across multiple delegations.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator


class TodoStatus(str, Enum):
    """Status of a todo item"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class TodoItem:
    """A todo item with metadata"""

    id: str
    title: str
    description: str
    status: TodoStatus
    created_at: str
    updated_at: str
    assigned_agent: Optional[str] = None
    priority: str = "medium"  # low, medium, high, critical
    dependencies: List[str] = None  # List of todo IDs this depends on
    notes: List[str] = None  # Progress notes

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.notes is None:
            self.notes = []


class TodoManager:
    """Manages todo items for the supervisor agent for a specific conversation"""

    def __init__(self, session_id: str):
        self.todos: Dict[str, TodoItem] = {}
        self.session_id = session_id

    def create_todo(
        self,
        title: str,
        description: str,
        priority: str = "medium",
        assigned_agent: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Create a new todo item"""
        todo_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        todo = TodoItem(
            id=todo_id,
            title=title,
            description=description,
            status=TodoStatus.PENDING,
            created_at=now,
            updated_at=now,
            assigned_agent=assigned_agent,
            priority=priority,
            dependencies=dependencies or [],
        )

        self.todos[todo_id] = todo
        return todo_id

    def update_todo_status(
        self, todo_id: str, status: TodoStatus, note: Optional[str] = None
    ) -> bool:
        """Update the status of a todo item"""
        if todo_id not in self.todos:
            return False

        self.todos[todo_id].status = status
        self.todos[todo_id].updated_at = datetime.now().isoformat()

        if note:
            self.todos[todo_id].notes.append(f"{datetime.now().isoformat()}: {note}")

        return True

    def add_note(self, todo_id: str, note: str) -> bool:
        """Add a progress note to a todo item"""
        if todo_id not in self.todos:
            return False

        self.todos[todo_id].notes.append(f"{datetime.now().isoformat()}: {note}")
        self.todos[todo_id].updated_at = datetime.now().isoformat()
        return True

    def get_todo(self, todo_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific todo item"""
        if todo_id not in self.todos:
            return None
        return asdict(self.todos[todo_id])

    def list_todos(
        self, status_filter: Optional[TodoStatus] = None
    ) -> List[Dict[str, Any]]:
        """List all todos, optionally filtered by status"""
        todos = list(self.todos.values())

        if status_filter:
            todos = [t for t in todos if t.status == status_filter]

        # Sort by priority and creation time
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        todos.sort(key=lambda x: (priority_order.get(x.priority, 2), x.created_at))

        return [asdict(todo) for todo in todos]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all todos"""
        status_counts = {}
        for status in TodoStatus:
            status_counts[status.value] = len(
                [t for t in self.todos.values() if t.status == status]
            )

        return {
            "session_id": self.session_id,
            "total_todos": len(self.todos),
            "status_counts": status_counts,
            "pending_todos": [
                asdict(t) for t in self.todos.values() if t.status == TodoStatus.PENDING
            ],
            "in_progress_todos": [
                asdict(t)
                for t in self.todos.values()
                if t.status == TodoStatus.IN_PROGRESS
            ],
            "completed_todos": [
                asdict(t)
                for t in self.todos.values()
                if t.status == TodoStatus.COMPLETED
            ],
        }


# Global registry of todo managers by conversation ID
_todo_managers_by_conv: Dict[str, TodoManager] = {}

def _get_todo_manager(conversation_id: str) -> TodoManager:
    """Get the current todo manager for this execution context, creating a new one if needed."""
    if conversation_id not in _todo_managers_by_conv:
        _todo_managers_by_conv[conversation_id] = TodoManager(conversation_id)
    return _todo_managers_by_conv[conversation_id]

def _format_current_todo_list(conversation_id: str) -> str:
    """Helper function to format the current todo list for display"""
    todo_manager = _get_todo_manager(conversation_id)
    current_todos = todo_manager.list_todos()

    result = "📋 **Current Todo List:**\n"
    if not current_todos:
        result += "No todos remaining.\n"
    else:
        status_emoji = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "cancelled": "❌",
            "blocked": "🚫",
        }
        priority_emoji = {"critical": "🔥", "high": "⚡", "medium": "📝", "low": "💤"}

        for todo in current_todos:
            emoji = status_emoji.get(todo["status"], "📝")
            p_emoji = priority_emoji.get(todo["priority"], "📝")
            result += f"{emoji} {p_emoji} **{todo['title']}** (ID: {todo['id']}) - {todo['status']}\n"

    return result


# Pydantic models for tool inputs
class CreateTodoInput(BaseModel):
    title: str = Field(description="Short title for the todo item")
    description: str = Field(description="Detailed description of the task")
    priority: str = Field(
        default="medium", description="Priority level: low, medium, high, critical"
    )
    assigned_agent: Optional[str] = Field(
        default=None,
        description="Agent type this task will be delegated to (e.g., 'think_execute')",
    )
    dependencies: Optional[List[str]] = Field(
        default=None,
        description="List of todo IDs (strings) that must be completed before this task. Must be an array/list, not a string. Example: ['abc123', 'def456'] or [] for no dependencies. Leave as null/empty if no dependencies.",
    )

    @field_validator("dependencies", mode="before")
    @classmethod
    def coerce_dependencies_to_list(cls, v):
        """Coerce string to list if a string is accidentally passed"""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        return [str(v)]


class UpdateTodoStatusInput(BaseModel):
    todo_id: str = Field(description="ID of the todo item to update")
    status: str = Field(
        description="New status: pending, in_progress, completed, cancelled, blocked"
    )
    note: Optional[str] = Field(default=None, description="Optional progress note")


class AddTodoNoteInput(BaseModel):
    todo_id: str = Field(description="ID of the todo item")
    note: str = Field(description="Progress note to add")


class GetTodoInput(BaseModel):
    todo_id: str = Field(description="ID of the todo item to retrieve")


class ListTodosInput(BaseModel):
    status_filter: Optional[str] = Field(
        default=None,
        description="Filter by status: pending, in_progress, completed, cancelled, blocked",
    )


def todo_management_tools(conversation_id: Optional[str] = None) -> List[StructuredTool]:
    """Create all todo management tools bound to a conversation_id"""
    
    if not conversation_id:
        return []

    def create_todo_tool(input_data: CreateTodoInput) -> str:
        """Create a new todo item for task tracking"""
        try:
            todo_manager = _get_todo_manager(conversation_id)
            todo_id = todo_manager.create_todo(
                title=input_data.title,
                description=input_data.description,
                priority=input_data.priority,
                assigned_agent=input_data.assigned_agent,
                dependencies=input_data.dependencies,
            )

            result = f"✅ Created todo item '{input_data.title}' with ID: {todo_id}\n\n"
            result += _format_current_todo_list(conversation_id)
            return result
        except Exception as e:
            return f"❌ Error creating todo: {str(e)}"

    def update_todo_status_tool(input_data: UpdateTodoStatusInput) -> str:
        """Update the status of a todo item"""
        try:
            try:
                status = TodoStatus(input_data.status.lower())
            except ValueError:
                return f"❌ Invalid status '{input_data.status}'. Valid statuses: {', '.join([s.value for s in TodoStatus])}"

            todo_manager = _get_todo_manager(conversation_id)
            success = todo_manager.update_todo_status(
                todo_id=input_data.todo_id, status=status, note=input_data.note
            )

            if success:
                status_emoji = {
                    TodoStatus.PENDING: "⏳",
                    TodoStatus.IN_PROGRESS: "🔄",
                    TodoStatus.COMPLETED: "✅",
                    TodoStatus.CANCELLED: "❌",
                    TodoStatus.BLOCKED: "🚫",
                }
                emoji = status_emoji.get(status, "📝")

                result = f"{emoji} Updated todo {input_data.todo_id} to status: {status.value}\n\n"
                result += _format_current_todo_list(conversation_id)
                return result
            else:
                return f"❌ Todo item {input_data.todo_id} not found"
        except Exception as e:
            return f"❌ Error updating todo: {str(e)}"

    def add_todo_note_tool(input_data: AddTodoNoteInput) -> str:
        """Add a progress note to a todo item"""
        try:
            todo_manager = _get_todo_manager(conversation_id)
            success = todo_manager.add_note(input_data.todo_id, input_data.note)

            if success:
                result = f"📝 Added note to todo {input_data.todo_id}\n\n"
                result += _format_current_todo_list(conversation_id)
                return result
            else:
                return f"❌ Todo item {input_data.todo_id} not found"
        except Exception as e:
            return f"❌ Error adding note: {str(e)}"

    def get_todo_tool(input_data: GetTodoInput) -> str:
        """Get details of a specific todo item"""
        try:
            todo_manager = _get_todo_manager(conversation_id)
            todo = todo_manager.get_todo(input_data.todo_id)

            if todo:
                status_emoji = {
                    "pending": "⏳",
                    "in_progress": "🔄",
                    "completed": "✅",
                    "cancelled": "❌",
                    "blocked": "🚫",
                }
                emoji = status_emoji.get(todo["status"], "📝")

                result = f"{emoji} **{todo['title']}** (ID: {todo['id']})\n"
                result += f"Status: {todo['status']}\n"
                result += f"Priority: {todo['priority']}\n"
                result += f"Description: {todo['description']}\n"

                if todo["assigned_agent"]:
                    result += f"Assigned to: {todo['assigned_agent']}\n"

                if todo["dependencies"]:
                    result += f"Dependencies: {', '.join(todo['dependencies'])}\n"

                if todo["notes"]:
                    result += "Notes:\n"
                    for note in todo["notes"][-3:]:  # Show last 3 notes
                        result += f"  - {note}\n"

                return result
            else:
                return f"❌ Todo item {input_data.todo_id} not found"
        except Exception as e:
            return f"❌ Error retrieving todo: {str(e)}"

    def list_todos_tool(input_data: ListTodosInput) -> str:
        """List all todo items, optionally filtered by status"""
        try:
            status_filter = None
            if input_data.status_filter:
                try:
                    status_filter = TodoStatus(input_data.status_filter.lower())
                except ValueError:
                    return f"❌ Invalid status filter '{input_data.status_filter}'. Valid statuses: {', '.join([s.value for s in TodoStatus])}"

            todo_manager = _get_todo_manager(conversation_id)
            todos = todo_manager.list_todos(status_filter)

            if not todos:
                filter_text = (
                    f" with status '{status_filter.value}'" if status_filter else ""
                )
                return f"📋 No todo items found{filter_text}"

            result = f"📋 **Todo List** ({len(todos)} items)\n\n"

            status_emoji = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "cancelled": "❌",
                "blocked": "🚫",
            }

            for todo in todos:
                emoji = status_emoji.get(todo["status"], "📝")
                priority_emoji = {
                    "critical": "🔥",
                    "high": "⚡",
                    "medium": "📝",
                    "low": "💤",
                }
                p_emoji = priority_emoji.get(todo["priority"], "📝")

                result += f"{emoji} {p_emoji} **{todo['title']}** (ID: {todo['id']})\n"
                result += f"   Status: {todo['status']} | Priority: {todo['priority']}\n"

                if todo["assigned_agent"]:
                    result += f"   Assigned: {todo['assigned_agent']}\n"

                result += f"   {todo['description'][:100]}{'...' if len(todo['description']) > 100 else ''}\n\n"

            return result
        except Exception as e:
            return f"❌ Error listing todos: {str(e)}"

    def get_todo_summary_tool() -> str:
        """Get a summary of all todo items"""
        try:
            todo_manager = _get_todo_manager(conversation_id)
            summary = todo_manager.get_summary()

            result = f"📊 **Todo Summary** (Session: {summary['session_id']})\n\n"
            result += f"Total todos: {summary['total_todos']}\n\n"

            status_emoji = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "cancelled": "❌",
                "blocked": "🚫",
            }

            result += "**Status Breakdown:**\n"
            for status, count in summary["status_counts"].items():
                emoji = status_emoji.get(status, "📝")
                result += f"{emoji} {status.title()}: {count}\n"

            if summary["in_progress_todos"]:
                result += f"\n**🔄 Currently In Progress ({len(summary['in_progress_todos'])}):**\n"
                for todo in summary["in_progress_todos"]:
                    result += f"- {todo['title']} (ID: {todo['id']})\n"

            if summary["pending_todos"]:
                result += f"\n**⏳ Pending ({len(summary['pending_todos'])}):**\n"
                for todo in summary["pending_todos"][:5]:
                    result += f"- {todo['title']} (ID: {todo['id']})\n"
                if len(summary["pending_todos"]) > 5:
                    result += f"... and {len(summary['pending_todos']) - 5} more\n"

            return result
        except Exception as e:
            return f"❌ Error getting summary: {str(e)}"

    return [
        StructuredTool.from_function(
            func=create_todo_tool,
            name="create_todo",
            description="Create a new todo item for task tracking. Use this to break down complex requests into manageable tasks. IMPORTANT: The 'dependencies' parameter must be a list/array of todo IDs (strings), not a single string. Use [] for no dependencies or ['todo_id1', 'todo_id2'] for multiple dependencies.",
            args_schema=CreateTodoInput,
        ),
        StructuredTool.from_function(
            func=update_todo_status_tool,
            name="update_todo_status",
            description="Update the status of a todo item (pending, in_progress, completed, cancelled, blocked). Use this to track progress.",
            args_schema=UpdateTodoStatusInput,
        ),
        StructuredTool.from_function(
            func=add_todo_note_tool,
            name="add_todo_note",
            description="Add a progress note to a todo item. Use this to record findings, issues, or updates.",
            args_schema=AddTodoNoteInput,
        ),
        StructuredTool.from_function(
            func=get_todo_tool,
            name="get_todo",
            description="Get detailed information about a specific todo item.",
            args_schema=GetTodoInput,
        ),
        StructuredTool.from_function(
            func=list_todos_tool,
            name="list_todos",
            description="List all todo items, optionally filtered by status. Use this to see current task state.",
            args_schema=ListTodosInput,
        ),
        StructuredTool.from_function(
            func=get_todo_summary_tool,
            name="get_todo_summary",
            description="Get a summary overview of all todo items and their status. Use this to understand overall progress.",
            args_schema=None,
        ),
    ]
