from math import factorial, e, log, exp, ceil

# Probability of Waiting (Optimized to Avoid Large Numbers)
def prob_waiting(traffic_intensity, num_agents):
    try:
        # Logarithmic calculation for numerator
        log_factorial = sum(log(i) for i in range(1, num_agents + 1))
        log_numerator = (
            num_agents * log(traffic_intensity)
            - log_factorial
            + log(num_agents)
            - log(num_agents - traffic_intensity)
        )
        numerator = exp(log_numerator)
    except OverflowError:
        # Handle edge cases with logarithmic calculations
        numerator = float("inf")
    
    # Iterative calculation for denominator
    y = 0.0
    term = 1.0  # Initial term (traffic_intensity^0 / 0!)
    for i in range(num_agents):
        y += term
        term *= traffic_intensity / (i + 1)  # Avoid direct computation of factorial

    return numerator / (y + numerator)

# Service Level
def compute_sla(pw, traffic_intensity, num_agents, targ_ans_time, aht):
    return 1 - (pw * (e ** -((num_agents - traffic_intensity) * (targ_ans_time / aht))))

# Adjusted Waiting Time Probability with Caller Patience
def prob_waiting_with_patience(pw, num_agents, traffic_intensity, caller_patience, aht_seconds):
    avg_wait_time = (pw * aht_seconds) / (num_agents - traffic_intensity)
    adjusted_pw = pw * (1 - e ** (-caller_patience / avg_wait_time)) if avg_wait_time > 0 else pw
    return adjusted_pw

# Function to calculate Erlang C outputs with abandonment limit
def get_erlang_c(volume, traffic_intensity, target_answer_time, aht_seconds, target_sla, shrinkage, caller_patience, max_abandonment):
    raw_agent = 1
    n = round(traffic_intensity + raw_agent)

    pw = prob_waiting(traffic_intensity, n)
    pw_adjusted = prob_waiting_with_patience(pw, n, traffic_intensity, caller_patience, aht_seconds)

    act_sla = compute_sla(pw_adjusted, traffic_intensity, n, target_answer_time, aht_seconds)
    percent_calls_abandoned = (pw - pw_adjusted) * 100

    while act_sla < target_sla or percent_calls_abandoned > max_abandonment:
        raw_agent += 1
        n = round(traffic_intensity + raw_agent)
        pw = prob_waiting(traffic_intensity, n)
        pw_adjusted = prob_waiting_with_patience(pw, n, traffic_intensity, caller_patience, aht_seconds)
        act_sla = compute_sla(pw, traffic_intensity, n, target_answer_time, aht_seconds)
        percent_calls_abandoned = (pw - pw_adjusted) * 100

    average_speed_of_answer = (pw * aht_seconds) / (n - traffic_intensity)

    percent_calls_answered_immediately = (1 - pw) * 100
    maximum_occupancy = (traffic_intensity / n) * 100
    n_shrinkage = n / (1 - shrinkage)

    return {
        'Volume': int(volume),
        'Traffic Intensity': int(traffic_intensity),
        'No. of Required Agents': int(n),
        'No. of Required Agents w/ Shrinkage': ceil(n_shrinkage),
        'Average Speed of Answer': round(average_speed_of_answer, 1),
        '% of Calls Answered Immediately': round(percent_calls_answered_immediately, 2),
        'Maximum Occupancy': round(maximum_occupancy, 2),
        'pct of Calls Abandoned': round(percent_calls_abandoned, 2),
        'SLA': round((act_sla * 100), 2)
    }
