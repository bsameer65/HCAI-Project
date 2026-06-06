# Responsible for:
    # selecting one penguin example,
    # selecting a desired target class,
    # generating random nearby examples,
    # checking which examples are predicted as the desired class,
    # ranking them by distance,
    # showing best counterfactuals.

# Important:
# Numerical features can be changed with random noise.

# Example:
# bill_length_mm + random noise

# Categorical features must be changed differently.
# Example:
# island can randomly become Biscoe, Dream, or Torgersen
# sex can become male or female

# Do not add decimal noise to categorical features.