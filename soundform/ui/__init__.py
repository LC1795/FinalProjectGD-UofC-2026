"""Desktop interface. The Matplotlib Qt backend is selected before any canvas is created."""

import matplotlib

matplotlib.use("QtAgg")
