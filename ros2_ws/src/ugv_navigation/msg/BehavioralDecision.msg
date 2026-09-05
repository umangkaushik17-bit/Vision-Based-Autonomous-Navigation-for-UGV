# BehavioralDecision.msg
# Tactical behavioral decision output from Milestone 13.10 BehavioralStateMachine.

std_msgs/Header header

# State: "PROCEED", "AVOID", "WAIT", "STOP"
string state

float32 command_v
float32 command_omega
string reason

# Context
uint32 blocking_obstacle_id
bool dynamic_collision
string dynamic_risk          # "FREE", "LOW", "MEDIUM", "HIGH", "LETHAL", "UNKNOWN_HORIZON"
bool unknown_horizon
float32 wait_elapsed_s
bool wait_timeout

# Deadlock signaling
bool deadlock
string deadlock_event        # "NONE", "DYNAMIC_DEADLOCK"

string transition
string previous_state
float32 state_dwell_time_s
float32 confidence
