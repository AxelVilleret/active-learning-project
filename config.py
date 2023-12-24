from selection_algorithms import *

### Active learning parameters
algorithms = {
    # RANDOM: select_samples_randomly,
    # CLUSTERING: select_samples_by_clustering,
    # REPRESENTATIVE_SAMPLING: select_samples_by_representative_sampling,
    # LEAST_CONFIDENCE: select_samples_by_least_confidence,
    # MARGIN: select_samples_by_margin,
    # ENTROPY: select_samples_by_entropy,
    # MIXED_WITH_LEAST_CONFIDENCE_AND_CLUSTERING: select_by_mixed_with_least_confidence_and_clustering,
    MIXED_WITH_MARGIN_AND_CLUSTERING: select_by_mixed_with_margin_and_clustering,
}

pourcentages = [
    1,
    5,
    10,
    15,
    20,
]

# pourcentages used to determine the number of samples to select
# pourcentages = [
#     1,
#     2,
#     4,
#     8,
#     16,
#     32,
# ]