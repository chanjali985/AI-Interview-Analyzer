import time
from collections import defaultdict, deque

    # Store request timestamps for each user
request_logs = defaultdict(lambda: deque())

MAX_REQUESTS = 20          # allowed requests
WINDOW_SIZE = 60           # 60 seconds

def is_allowed(user_id):
    now = time.time()
    q = request_logs[user_id]

    # Remove timestamps older than 60 seconds
    while q and now - q[0] > WINDOW_SIZE:
        q.popleft()

    # Check if within limit
    if len(q) < MAX_REQUESTS:
        q.append(now)
        return True
    else:
        return False


    # Example usage:
user = "user_123"

for i in range(25):
    if is_allowed(user):
        print(i, "→ Request allowed")
    else:
        print(i, "→ Rate limit exceeded")
