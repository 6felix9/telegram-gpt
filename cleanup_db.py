#!/usr/bin/env python3
"""
Database cleanup utility for removing unauthorized user history.

This script removes:
1. All messages in private chats with unauthorized users
2. Individual messages from unauthorized users in group chats

Creates automatic backup before deletion.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import Config
from database import Database
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preview_cleanup(db: Database, authorized_user_id: str) -> dict:
    """
    Preview what will be deleted without actually deleting.

    Args:
        db: Database instance
        authorized_user_id: Main authorized user ID

    Returns:
        Statistics about what would be deleted
    """
    # Get authorized user IDs
    authorized_ids = {authorized_user_id}
    granted_users = db.get_granted_users()
    for user_id, _ in granted_users:
        authorized_ids.add(user_id)

    # Count unauthorized users with history
    with db._get_connection() as conn:
        placeholders = ",".join("?" * len(authorized_ids))
        cursor = conn.execute(
            f"""
            SELECT DISTINCT user_id
            FROM messages
            WHERE user_id IS NOT NULL
            AND user_id NOT IN ({placeholders})
            """,
            tuple(authorized_ids),
        )
        unauthorized_user_ids = [row["user_id"] for row in cursor]

        if not unauthorized_user_ids:
            return {
                "unauthorized_users": 0,
                "private_chats": 0,
                "private_messages": 0,
                "group_messages": 0,
                "total_messages": 0,
                "user_ids": [],
            }

        unauth_placeholders = ",".join("?" * len(unauthorized_user_ids))

        # Count private chat messages
        cursor = conn.execute(
            f"""
            SELECT COUNT(*) as count, COUNT(DISTINCT chat_id) as chats
            FROM messages
            WHERE chat_id IN ({unauth_placeholders})
            """,
            tuple(unauthorized_user_ids),
        )
        row = cursor.fetchone()
        private_messages = row["count"]
        private_chats = row["chats"]

        # Count group chat messages
        cursor = conn.execute(
            f"""
            SELECT COUNT(*) as count
            FROM messages
            WHERE user_id IS NOT NULL
            AND user_id IN ({unauth_placeholders})
            AND chat_id NOT IN ({unauth_placeholders})
            """,
            tuple(unauthorized_user_ids) + tuple(unauthorized_user_ids),
        )
        group_messages = cursor.fetchone()["count"]

        return {
            "unauthorized_users": len(unauthorized_user_ids),
            "private_chats": private_chats,
            "private_messages": private_messages,
            "group_messages": group_messages,
            "total_messages": private_messages + group_messages,
            "user_ids": unauthorized_user_ids,
        }


def main():
    """Main entry point for cleanup script."""
    print("=" * 70)
    print("Database Cleanup Utility - Remove Unauthorized User History")
    print("=" * 70)
    print()

    # Validate configuration
    try:
        Config.validate()
    except SystemExit:
        logger.error("Configuration validation failed. Please check your .env file.")
        return 1

    # Initialize database
    try:
        db = Database(Config.DATABASE_PATH)
        logger.info(f"Connected to database: {Config.DATABASE_PATH}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    # Show authorized users
    print(f"Main authorized user: {Config.AUTHORIZED_USER_ID}")
    granted_users = db.get_granted_users()
    if granted_users:
        print(f"Granted users: {len(granted_users)}")
        for user_id, granted_at in granted_users:
            print(f"  - {user_id} (granted: {granted_at})")
    else:
        print("Granted users: None")
    print()

    # Preview cleanup
    print("Analyzing database...")
    try:
        stats = preview_cleanup(db, Config.AUTHORIZED_USER_ID)
    except Exception as e:
        logger.error(f"Failed to analyze database: {e}", exc_info=True)
        return 1

    if stats["total_messages"] == 0:
        print("✅ No unauthorized user history found. Database is clean!")
        return 0

    # Show what will be deleted
    print("=" * 70)
    print("Preview: The following data will be DELETED:")
    print("=" * 70)
    print(f"Unauthorized users found:     {stats['unauthorized_users']}")
    print(f"Private chats to delete:      {stats['private_chats']}")
    print(f"Private chat messages:        {stats['private_messages']}")
    print(f"Group chat messages:          {stats['group_messages']}")
    print(f"Total messages to delete:     {stats['total_messages']}")
    print()
    print("Unauthorized user IDs:")
    for user_id in stats["user_ids"]:
        print(f"  - {user_id}")
    print()

    # Confirm deletion
    print("⚠️  WARNING: This action cannot be undone!")
    print("A backup will be created automatically before deletion.")
    print()

    try:
        response = input("Do you want to proceed? (yes/no): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled by user.")
        return 1

    if response != "yes":
        print("Operation cancelled.")
        return 0

    # Create backup
    print()
    print("Creating backup...")
    try:
        backup_path = db.backup_database()
        if backup_path:
            print(f"✅ Backup created: {backup_path}")
        else:
            print("⚠️  Warning: Backup creation failed, but continuing...")
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        print("⚠️  Warning: Backup creation failed!")
        response = input("Continue anyway? (yes/no): ").strip().lower()
        if response != "yes":
            print("Operation cancelled.")
            return 1

    # Perform cleanup
    print()
    print("Performing cleanup...")
    try:
        result = db.cleanup_unauthorized_history(Config.AUTHORIZED_USER_ID)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        print(f"❌ Cleanup failed: {e}")
        print(f"You can restore from backup: {backup_path if 'backup_path' in locals() else 'N/A'}")
        return 1

    # Show results
    print()
    print("=" * 70)
    print("Cleanup Complete!")
    print("=" * 70)
    print(f"Private chats deleted:        {result['private_chats_deleted']}")
    print(f"Group messages deleted:       {result['group_messages_deleted']}")
    print(f"Total messages deleted:       {result['total_messages_deleted']}")
    print(f"Unauthorized users cleaned:   {len(result['unauthorized_user_ids'])}")
    print()
    print("✅ Database cleanup successful!")

    if 'backup_path' in locals():
        print(f"Backup available at: {backup_path}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
