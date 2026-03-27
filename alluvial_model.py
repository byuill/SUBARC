import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# =====================================================================
# USER CONTROL PANEL
# =====================================================================
VALLEY_WIDTH = 4500.0                # Total width of the valley cross-section (m)
GRID_RESOLUTION_X = 10.0             # Horizontal cell size (m)
GRID_RESOLUTION_Y = 0.01             # Vertical cell size (m) for tracking layers
BANKFULL_WIDTH = 60.0                # Channel width (m)
BANKFULL_DEPTH = 3.0                 # Channel depth (m)
CHANNEL_AGGRADATION_RATE = 0.004     # Aggradation rate per annual flood (m)
MAX_TIME = 11000                     # Total simulation time (years)
PLOT_MODE = 'animation'             # 'final_only' or 'animation'
ANIMATION_FILENAME = 'alluvial_architecture.gif'
FINAL_PLOT_FILENAME = 'alluvial_architecture_output.png'
# =====================================================================

class AlluvialModel:
    def __init__(self, valley_width=VALLEY_WIDTH, grid_resolution_x=GRID_RESOLUTION_X, grid_resolution_y=GRID_RESOLUTION_Y,
                 bankfull_width=BANKFULL_WIDTH, bankfull_depth=BANKFULL_DEPTH, channel_aggradation_rate=CHANNEL_AGGRADATION_RATE,
                 max_time=MAX_TIME):
        """
        Initialize the alluvial architecture model.
        Default values based on Trinity River near Dallas, TX (X-sec A).
        """
        self.valley_width = valley_width
        self.dx = grid_resolution_x
        self.dy = grid_resolution_y
        self.nx = int(self.valley_width / self.dx)

        self.bankfull_width = bankfull_width
        self.bankfull_depth = bankfull_depth
        self.channel_cells = int(self.bankfull_width / self.dx)

        self.ach = channel_aggradation_rate # Channel aggradation rate per annual flood
        self.max_time = max_time

        # Grid to store elevation at each x-coordinate
        # Initial topography: flat bedrock at y=0
        self.elevation = np.zeros(self.nx)

        # Store stratigraphy: List of arrays or similar structure.
        # Given potential large depth, we can store it efficiently as an array of layers for each column
        # Or store it as a list of (type, thickness) for each column.
        # Alternatively, a 2D array if max depth is known. Let's start with a dynamic list of lists
        # 0 = Bedrock, 1 = Clay/Silt (Floodplain), 2 = Sand (Channel)
        self.stratigraphy = [[] for _ in range(self.nx)]

        # Initial channel location
        # Place channel near the middle initially
        self.channel_idx = self.nx // 2

        # State variables for meandering
        self.meander_amplitude = 0.0
        self.meander_direction = random.choice([-1, 1])
        self.cbw_avg = 800.0 # Average channel belt width (~800m based on text)

        # Add initial bedrock layer to stratigraphy (e.g., 0 depth, just base)

    def migrate_channel(self):
        """
        Simulate channel lateral migration (meander sequence).
        Calculate bank erosion rate based on meander amplitude (proxy for radius of curvature).
        """
        # Bank erosion rate function (based on ERDC TR-13-5 Fig 3, approximated as quadratic or parabolic)
        # Assuming maximum migration rate at MW / CBW_AVG ~ 0.5
        relative_amp = self.meander_amplitude / self.cbw_avg

        # Approximate erosion rate: peaks at relative_amp=0.5, drops to ~0 at 1.0
        # This is an empirical estimation based on text
        # Let's say base max erosion rate is 2 m/year
        max_erosion_rate = 2.0
        erosion_rate = max_erosion_rate * (1.0 - 4.0 * (relative_amp - 0.5)**2)
        if erosion_rate < 0.1:
            erosion_rate = 0.1

        # Time to migrate one grid cell (dx)
        dt = self.dx / erosion_rate

        # Update channel position
        self.channel_idx += self.meander_direction
        self.meander_amplitude += self.dx

        # Check boundary reflection (valley walls)
        if self.channel_idx - self.channel_cells//2 < 0:
            self.channel_idx = self.channel_cells//2
            self.meander_direction *= -1
            self.meander_amplitude = 0.0 # terminate meander
        elif self.channel_idx + self.channel_cells//2 >= self.nx:
            self.channel_idx = self.nx - 1 - self.channel_cells//2
            self.meander_direction *= -1
            self.meander_amplitude = 0.0

        return dt

    def calculate_flood_hydrology(self, dt):
        """
        Sample maximum flood discharge over time step `dt` using Log Pearson Type III.
        Calculate number of floods exceeding bankfull.
        """
        # Bankfull discharge base assumption (approximate Q for 2-year flood ~ 600 m3/s)
        # We assume a log-normal or Weibull distribution
        # Let's say mean peak annual flood is 400 m3/s, std dev 200 m3/s
        # Number of years in this time step
        years = int(dt)
        if years < 1:
            years = 1

        floods = np.random.lognormal(mean=6.0, sigma=0.5, size=years)
        max_flood = np.max(floods)

        bankfull_discharge = 600.0
        floods_exceeding = np.sum(floods > bankfull_discharge)

        # Max lateral extent of floodplain deposition w_max
        # For a simplified model, if it exceeds bankfull, let's say it scales linearly or logarithmically
        if max_flood > bankfull_discharge:
            w_max = self.bankfull_width * (max_flood / bankfull_discharge)
            # Cap at valley width
            if w_max > self.valley_width:
                w_max = self.valley_width
        else:
            w_max = 0.0

        return max_flood, floods_exceeding, w_max

    def apply_sedimentation(self, floods_exceeding, w_max):
        """
        Apply channel aggradation and floodplain deposition.
        Floodplain deposition follows ai = aCH * exp(-b * (wi / wMAX))
        """
        if floods_exceeding == 0 or w_max == 0:
            return

        # Total channel aggradation for this time step
        total_ach = self.ach * floods_exceeding

        # Exponential decay constant b
        b = 3.0

        # Channel margins
        left_margin = self.channel_idx - self.channel_cells // 2
        right_margin = self.channel_idx + self.channel_cells // 2

        # Apply sedimentation across the grid
        for i in range(self.nx):
            if left_margin <= i <= right_margin:
                # Within channel: aggrade sand (type 2)
                self.elevation[i] += total_ach
                self.stratigraphy[i].append((2, total_ach))
            else:
                # Floodplain: aggrade silt/clay (type 1)
                # Calculate distance from nearest channel margin
                if i < left_margin:
                    wi = (left_margin - i) * self.dx
                else:
                    wi = (i - right_margin) * self.dx

                if wi <= w_max:
                    ai = total_ach * np.exp(-b * (wi / w_max))
                    self.elevation[i] += ai
                    self.stratigraphy[i].append((1, ai))

    def trigger_avulsion(self, max_flood):
        """
        Check conditions for process-based avulsion and execute if triggered.
        Process-based avulsion requires river in flood stage and superelevation > 0.
        Relocate to the lowest point in the floodplain.
        """
        bankfull_discharge = 600.0
        # River must be in flood stage
        if max_flood <= bankfull_discharge:
            return False

        # Minimum floodplain elevation
        min_fp_elev = np.min(self.elevation)
        channel_elev = self.elevation[self.channel_idx]

        # Superelevation
        superelevation = channel_elev - min_fp_elev
        if superelevation <= 0:
            return False

        # Superelevation relative to bankfull depth (SE*)
        se_star = superelevation / self.bankfull_depth

        # Probability based on SE* (approximating Figure 6 in ERDC report)
        # Probability increases rapidly as SE* approaches 2 or 3
        if se_star <= 0:
            prob = 0.0
        elif se_star < 1.0:
            prob = 0.01 * se_star
        elif se_star < 2.0:
            prob = 0.1 * (se_star - 1.0) + 0.01
        else:
            prob = 0.5 * (se_star - 2.0) + 0.1

        prob = min(prob, 1.0)

        # Trigger avulsion?
        if random.random() < prob:
            # Find the index of the minimum elevation
            min_indices = np.where(self.elevation == min_fp_elev)[0]
            new_channel_idx = random.choice(min_indices)

            # Ensure it's not too close to the edge
            if new_channel_idx < self.channel_cells // 2:
                new_channel_idx = self.channel_cells // 2
            elif new_channel_idx >= self.nx - self.channel_cells // 2:
                new_channel_idx = self.nx - 1 - self.channel_cells // 2

            self.channel_idx = new_channel_idx
            self.meander_amplitude = 0.0 # Reset meander sequence
            return True

        return False

    def apply_subsidence(self, dt):
        """
        Simulate compaction of fine sediments over time step `dt`.
        Max subsidence rate ~ 0.5 mm/year = 0.0005 m/year.
        """
        max_subsidence_rate = 0.0005
        max_sub = max_subsidence_rate * dt

        # Simple implementation: subside the whole elevation uniformly based on the amount of silt/clay (type 1)
        # More accurately, it depends on the column of fine sediment
        for i in range(self.nx):
            # Calculate total thickness of fine sediment
            fine_thickness = sum([thickness for stype, thickness in self.stratigraphy[i] if stype == 1])

            if fine_thickness > 0:
                # Subside proportional to fine sediment thickness (up to max_sub)
                # Assumed that e.g. 10m of mud subsides max_sub
                # This is a highly simplified approach to capture the physical effect
                compaction = min(max_sub * (fine_thickness / 10.0), max_sub)

                # Apply compaction to elevation
                self.elevation[i] -= compaction

                # We do not subtract from stratigraphy thicknesses to keep original deposition records,
                # though technically the layers themselves thin out.

if __name__ == "__main__":
    import time

    print("Initializing Alluvial Architecture Model...")
    model = AlluvialModel(max_time=MAX_TIME)

    current_time = 0.0

    print(f"Running simulation for {model.max_time} years...")
    start_time = time.time()

    # Store history for animation
    history_time = []
    history_elevation = []
    history_sand = []
    history_silt = []

    save_interval = max(1, model.max_time // 100) # Save ~100 frames for animation
    next_save_time = 0.0

    while current_time < model.max_time:
        # 1. Meander and determine time step
        dt = model.migrate_channel()
        current_time += dt

        # 2. Compute hydrology for this time step
        max_flood, floods_exceeding, w_max = model.calculate_flood_hydrology(dt)

        # 3. Apply sedimentation
        model.apply_sedimentation(floods_exceeding, w_max)

        # 4. Check for avulsions
        model.trigger_avulsion(max_flood)

        # 5. Apply subsidence
        model.apply_subsidence(dt)

        if current_time >= next_save_time and PLOT_MODE == 'animation':
            # Compute current thickness for animation frame
            sand_th = np.zeros(model.nx)
            silt_th = np.zeros(model.nx)
            for i in range(model.nx):
                sand_th[i] = sum([thickness for stype, thickness in model.stratigraphy[i] if stype == 2])
                silt_th[i] = sum([thickness for stype, thickness in model.stratigraphy[i] if stype == 1])
            history_time.append(current_time)
            history_elevation.append(np.copy(model.elevation))
            history_sand.append(sand_th)
            history_silt.append(silt_th)
            next_save_time += save_interval

    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")

    x_coords = np.arange(model.nx) * model.dx

    if PLOT_MODE == 'final_only':
        final_elevation = model.elevation
        sand_thickness = np.zeros(model.nx)
        silt_thickness = np.zeros(model.nx)

        for i in range(model.nx):
            sand_thickness[i] = sum([thickness for stype, thickness in model.stratigraphy[i] if stype == 2])
            silt_thickness[i] = sum([thickness for stype, thickness in model.stratigraphy[i] if stype == 1])

        plt.figure(figsize=(10, 6))
        plt.fill_between(x_coords, 0, silt_thickness, color='tan', label='Floodplain Deposits (Silt/Clay)')
        plt.fill_between(x_coords, silt_thickness, silt_thickness + sand_thickness, color='orange', label='Channel Deposits (Sand)')
        plt.plot(x_coords, final_elevation, color='black', label='Final Topography')

        plt.title('Simulated Alluvial Architecture')
        plt.xlabel('Cross-valley Distance (m)')
        plt.ylabel('Elevation / Thickness (m)')
        plt.legend()
        plt.grid(True)
        plt.savefig(FINAL_PLOT_FILENAME)
        print(f"Saved output plot to '{FINAL_PLOT_FILENAME}'.")

    elif PLOT_MODE == 'animation':
        print("Generating animation...")
        fig, ax = plt.subplots(figsize=(10, 6))

        def update(frame):
            ax.clear()
            silt_th = history_silt[frame]
            sand_th = history_sand[frame]
            elev = history_elevation[frame]

            ax.fill_between(x_coords, 0, silt_th, color='tan', label='Floodplain Deposits (Silt/Clay)')
            ax.fill_between(x_coords, silt_th, silt_th + sand_th, color='orange', label='Channel Deposits (Sand)')
            ax.plot(x_coords, elev, color='black', label='Topography')

            # Keep axes stable
            max_elev = np.max(history_elevation[-1]) * 1.1
            ax.set_ylim(0, max_elev if max_elev > 0 else 10)
            ax.set_xlim(0, VALLEY_WIDTH)

            ax.set_title(f"Simulated Alluvial Architecture (Time: {history_time[frame]:.0f} years)")
            ax.set_xlabel("Cross-valley Distance (m)")
            ax.set_ylabel("Elevation / Thickness (m)")
            ax.legend(loc='upper right')
            ax.grid(True)

        ani = animation.FuncAnimation(fig, update, frames=len(history_time), repeat=False)
        ani.save(ANIMATION_FILENAME, writer='pillow', fps=10)
        print(f"Saved animation to '{ANIMATION_FILENAME}'.")
