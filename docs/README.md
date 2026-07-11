# A/B Testing

## What is an A/B Test?

A way to determine which variant of something performs better. This is done by dividing a set of people into two groups, showing them different versions and analysing the results based on a specific goal to determine the better performing one. 

## Types of A/B Tests

### One Sample A/B Test

- Compare two variants of the same model.
- Compares a new model against a baseline model. Text vs Control.

### Two Sample A/B Test

- Compare the performance of two different models.

## Steps

1. Define the hypothesis: "The new ML-based recommendation system will increase the click-through rate (CTR) by at least 15% compared to the existing rule-based system". H0: The old system will perform better, H1: The new system will perform better.
2. Define success metric:  Click-Through Rate (CTR) = Number of Clicks / Number of Impressions, Conversion rate (CR) = Number of Purchases / Number of Clicks
3. Calculate sample size and duration: MDe (Minimum Detectable Effect) is the smallest change we care about detecting. If you wanted to detect a 0.0001% change in a metric on a site with 100,000 monthly visitors you would require 10 months and 1 million users to detect the change. 
```Python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

def calculate_sample_size(baseline_conversion, mde, power=0.8, significance_level=0.05):
   expected_conversion = baseline_conversion * (1 + mde)
  
   z_alpha = stats.norm.ppf(1 - significance_level/2)
   z_beta = stats.norm.ppf(power)
  
   sd1 = np.sqrt(baseline_conversion * (1 - baseline_conversion))
   sd2 = np.sqrt(expected_conversion * (1 - expected_conversion))
  
   numerator = (z_alpha * np.sqrt(2 * sd1**2) + z_beta * np.sqrt(sd1**2 + sd2**2))**2
   denominator = (expected_conversion - baseline_conversion)**2
  
   sample_size_per_variant = np.ceil(numerator / denominator)
  
   return int(sample_size_per_variant)

def calculate_experiment_duration(sample_size_per_variant, daily_visitors, traffic_allocation=0.5):
   visitors_per_variant_per_day = daily_visitors * traffic_allocation / 2
   days_required = np.ceil(sample_size_per_variant / visitors_per_variant_per_day)
  
   return int(days_required)

# Example MDE/sample size tradeoff for Jean's website
daily_visitors = 100000 / 30  # Convert monthly to daily visitors
baseline_conversion = 0.05    # Jean's current landing page CTR (baseline conv rate of 5%)

# Create a table of sample sizes for different MDEs
mde_values = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15]  # 1% to 15% change
traffic_allocations = [0.1, 0.5, 1.0]  # 10%, 50%, and 100% of website traffic

results = []
for mde in mde_values:
   sample_size = calculate_sample_size(baseline_conversion, mde)
  
   for allocation in traffic_allocations:
       duration = calculate_experiment_duration(sample_size, daily_visitors, allocation)
       results.append({
           'MDE': f"{mde*100:.1f}%",
           'Traffic Allocation': f"{allocation*100:.0f}%",
           'Sample Size per Variant': f"{sample_size:,}",
           'Duration (days)': duration
       })

# Create a DataFrame and display the results
df_results = pd.DataFrame(results)
print("Sample Size and Duration for Different MDEs:")
print(df_results)

# Visualize the relationship between MDE and sample size
plt.figure(figsize=(10, 6))
mde_range = np.arange(0.01, 0.2, 0.01)
sample_sizes = [calculate_sample_size(baseline_conversion, mde) for mde in mde_range]

plt.plot(mde_range * 100, sample_sizes)
plt.xlabel('Minimum Detectable Effect (%)')
plt.ylabel('Required Sample Size per Variant')
plt.title('Required Sample Size vs. MDE')
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.savefig('sample_size_vs_mde.png')
plt.show()
```
4. Collect data
5. Analyze results: 
```python
!pip install numpy scipy

import numpy as np
import scipy.stats as stats

cc = 1200  # control clicks
ci = 10000  # control impressions

tc = 1500  # test clicks
ti = 10000  # test impressions

ctr_c = cc / ci
ctr_t = tc / ti

table = np.array([[cc, ci - cc],
                  [tc, ti - tc]])

chi2, p, _, _ = stats.chi2_contingency(table)

print(f"Control CTR: {ctr_c:.2%}")
print(f"Test CTR: {ctr_t:.2%}")
print(f"Chi-Square Test p-value: {p:.5f}")

if p < 0.05:
    print("The difference is statistically significant.Implement the new recommendation system.")
else:
    print("No significant difference. Further testing needed.")
```

> Control CTR: 12.00% 
Test CTR: 15.00% 
Chi-Square Test p-value: 0.00000 
The difference is statistically significant. Implement the new recommendation system.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def analyze_ab_test_results(control_visitors, control_conversions,
                          treatment_visitors, treatment_conversions,
                          significance_level=0.05):

   # Calculate conversion rates
   control_rate = control_conversions / control_visitors
   treatment_rate = treatment_conversions / treatment_visitors
  
   # Calculate absolute and relative differences
   absolute_diff = treatment_rate - control_rate
   relative_diff = absolute_diff / control_rate
  
   # Calculate standard errors
   control_se = np.sqrt(control_rate * (1 - control_rate) / control_visitors)
   treatment_se = np.sqrt(treatment_rate * (1 - treatment_rate) / treatment_visitors)
  
   # Calculate z-score
   pooled_se = np.sqrt(control_se**2 + treatment_se**2)
   z_score = absolute_diff / pooled_se
  
   # Calculate p-value (two-tailed test)
   p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
  
   # Calculate confidence interval
   z_critical = stats.norm.ppf(1 - significance_level/2)
   margin_of_error = z_critical * pooled_se
   ci_lower = absolute_diff - margin_of_error
   ci_upper = absolute_diff + margin_of_error
  
   # Determine if result is statistically significant
   is_significant = p_value < significance_level
  
   return {
       'control_rate': control_rate,
       'treatment_rate': treatment_rate,
       'absolute_diff': absolute_diff,
       'relative_diff': relative_diff * 100,  # Convert to percentage
       'z_score': z_score,
       'p_value': p_value,
       'ci_lower': ci_lower,
       'ci_upper': ci_upper,
       'is_significant': is_significant
   }
```

6. Make a decision: if p < 0.05 reject null hypothesis, replace new system with old system. If p > 0.05 accept null hypothesis results are inconclusive, may require further tests.

## References

1. (Ayushi Mahariye, GeeksForGeeks, https://www.geeksforgeeks.org/data-science/a-b-testing-using-python/)
2. (Natassha Selvaraj, KDnuggets, https://www.kdnuggets.com/a-complete-guide-to-a-b-testing-in-python)