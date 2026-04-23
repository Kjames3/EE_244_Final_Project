import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/thor_magni_processed/THOR-Magni_170522_SC1B_R2_features.csv")

print("=== Data Sample ===")
print(df.head(10))

print("\n=== Stats ===")
print(df[['rel_x','rel_y','vx','vy','speed']].describe())

# Plot trajectories of all bodies
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for body, group in df.groupby('body'):
    axes[0].plot(group['rel_x'], group['rel_y'], 
                 alpha=0.6, linewidth=0.8, label=body)

axes[0].set_title('Pedestrian Trajectories (relative to robot)')
axes[0].set_xlabel('X position (m)')
axes[0].set_ylabel('Y position (m)')
axes[0].legend(fontsize=7)
axes[0].set_aspect('equal')
axes[0].grid(True)

# Velocity distribution
axes[1].hist(df['speed'].clip(0, 3), bins=50, 
             color='steelblue', edgecolor='white')
axes[1].set_title('Speed Distribution (m/s)')
axes[1].set_xlabel('Speed (m/s)')
axes[1].set_ylabel('Count')
axes[1].grid(True)

plt.tight_layout()
plt.savefig("dataset/thor_magni_processed/sanity_check.png", dpi=150)
plt.show()
print("\nPlot saved.")