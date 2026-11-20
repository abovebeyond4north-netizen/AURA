"""Protected logical budget caps; not operating-system resource isolation."""
MAX_STEPS=3000
MAX_REPLICATES=20

def validate_budget(training_steps,replicates):
    if type(training_steps) is not int or not 5<=training_steps<=MAX_STEPS:
        raise ValueError('Invalid evaluation steps')
    if type(replicates) is not int or not 1<=replicates<=MAX_REPLICATES:
        raise ValueError('Invalid replicate count')
