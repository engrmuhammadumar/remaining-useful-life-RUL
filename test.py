{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c3ddecaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Step 1: Create a graph\n",
    "G = nx.Graph()\n",
    "\n",
    "# Step 2: Add nodes (dots)\n",
    "G.add_nodes_from([\"Alice\", \"Bob\", \"Charlie\"])\n",
    "\n",
    "# Step 3: Add edges (connections)\n",
    "G.add_edge(\"Alice\", \"Bob\")\n",
    "G.add_edge(\"Bob\", \"Charlie\")\n",
    "\n",
    "# Step 4: Draw it\n",
    "nx.draw(G, with_labels=True, node_color='lightblue', node_size=1500, font_size=15)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c356216b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "# Simulated features (100 samples)\n",
    "np.random.seed(42)\n",
    "vibration = np.linspace(0.2, 1.0, 100) + np.random.normal(0, 0.05, 100)\n",
    "temperature = np.linspace(30, 80, 100) + np.random.normal(0, 1.0, 100)\n",
    "acoustic = np.linspace(0.1, 0.9, 100) + np.random.normal(0, 0.03, 100)\n",
    "\n",
    "# Combine features\n",
    "X = np.vstack([vibration, temperature, acoustic]).T\n",
    "\n",
    "# Simulated RUL (depends mostly on vibration + temperature)\n",
    "RUL = 150 - 60*vibration - 0.8*temperature + np.random.normal(0, 3, 100)\n",
    "\n",
    "# Train Random Forest\n",
    "rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)\n",
    "rf.fit(X, RUL)\n",
    "\n",
    "# Get feature importance\n",
    "features = ['Vibration', 'Temperature', 'Acoustic']\n",
    "importance = rf.feature_importances_\n",
    "\n",
    "# Plot\n",
    "plt.figure(figsize=(7,4))\n",
    "plt.barh(features, importance, color='skyblue')\n",
    "plt.xlabel('Importance Score')\n",
    "plt.title('Feature Importance in Random Forest (RUL Prediction)')\n",
    "plt.grid(True, axis='x')\n",
    "plt.show()\n",
    "\n",
    "for f, imp in zip(features, importance):\n",
    "    print(f\"{f}: {imp:.3f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87d27d14",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "langchainenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
