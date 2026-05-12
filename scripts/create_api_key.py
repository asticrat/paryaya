#!/usr/bin/env python3
"""
Create and list Paryaya API keys in Redis.

Usage:
    python scripts/create_api_key.py --company "Nepal Telecom" --plan business
    python scripts/create_api_key.py --list

Environment:
    REDIS_URL          — Redis connection string (default: redis://localhost:6379/0)
    ADMIN_SECRET_KEY   — required when creating / listing (optional for local use)
"""
import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timezone

import redis


def _redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except redis.ConnectionError as e:
        print(f"❌  Cannot connect to Redis at {url}: {e}", file=sys.stderr)
        sys.exit(1)


def create_key(company: str, plan: str) -> str:
    valid_plans = {"starter", "business", "enterprise"}
    if plan not in valid_plans:
        print(f"❌  Invalid plan '{plan}'. Choose from: {sorted(valid_plans)}", file=sys.stderr)
        sys.exit(1)

    api_key = f"sk-paryaya-{secrets.token_hex(16)}"
    record  = {
        "company":        company,
        "plan":           plan,
        "usage_minutes":  0.0,
        "requests_today": 0,
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }

    r = _redis_client()
    r.set(f"key:{api_key}", json.dumps(record))
    print(api_key)
    return api_key


def list_keys() -> None:
    r = _redis_client()
    keys = list(r.scan_iter("key:sk-paryaya-*"))

    if not keys:
        print("No API keys found.")
        return

    # Header
    print(f"\n{'API Key':50}  {'Company':20}  {'Plan':12}  {'Usage (min)':>12}  {'Req Today':>10}  Created")
    print("─" * 130)

    for redis_key in sorted(keys):
        raw = r.get(redis_key)
        if not raw:
            continue
        data    = json.loads(raw)
        api_key = redis_key.removeprefix("key:")
        print(
            f"{api_key:50}  "
            f"{data.get('company', '?'):20}  "
            f"{data.get('plan', '?'):12}  "
            f"{data.get('usage_minutes', 0):>12.2f}  "
            f"{data.get('requests_today', 0):>10}  "
            f"{data.get('created_at', '?')[:19]}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Paryaya API keys")
    parser.add_argument("--company", help="Company name (required for creation)")
    parser.add_argument("--plan",    help="Plan: starter | business | enterprise")
    parser.add_argument("--list",    action="store_true", help="List all keys with usage stats")
    args = parser.parse_args()

    if args.list:
        list_keys()
    elif args.company and args.plan:
        create_key(args.company, args.plan)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
