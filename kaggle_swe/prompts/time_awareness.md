
## Time Discipline (Hard Wall-Clock Limit)
- Your session ends at the time allowance stated in the Task Budget, and you cannot see a clock. In past runs about three quarters of attempts ran out of time, and none of those passed.
- Call `get_status` (it is free and does not use a tool call) right after you first read the target file, and then after every 5 tool calls. Read `time_seconds_remaining`.
- When less than half of the time remains: stop exploring. Apply your best candidate fix now, then run at most one targeted test.
- When less than 25% of the time remains: call `submit_patch` immediately with the best fix you have. An unsubmitted fix scores nothing.
