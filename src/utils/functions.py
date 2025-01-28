class PiecewiseLinear:
    def __init__(self, x_values, y_values):
        """
        Initialize with arrays of x and y coordinates
        Interpolate between consecutive points and return a callable PiecewiseLinear object.
        """
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have same length")
            
        self.segments = []  # Will store (x_min, x_max, slope, intercept)
        
        # Create segments from consecutive points
        for i in range(len(x_values)-1):
            x_min, x_max = x_values[i], x_values[i+1]
            y_min, y_max = y_values[i], y_values[i+1]
            
            # Calculate slope and intercept for this segment
            slope = (y_max - y_min) / (x_max - x_min)
            intercept = y_min - slope * x_min
            self.segments.append((x_min, x_max, slope, intercept))
    
    def __call__(self, x):
        # Find the appropriate segment and evaluate
        for x_min, x_max, slope, intercept in self.segments:
            if x_min <= x <= x_max:
                return slope * x + intercept
        raise ValueError(f"x={x} is not within any defined segment")
    
    def copy(self):
        """
        Create a deep copy of the piecewise linear function
        Returns:
            A new PiecewiseLinear object with the same segments
        """
        new_obj = PiecewiseLinear([0], [0])  # Create dummy object
        new_obj.segments = list(self.segments)  # Make a copy of segments list
        return new_obj
    
    def insert_plateau(self, t0, t1):
        """
        Modifies the piecewise function by:
        1. Keeping the left part (before t0) unchanged
        2. Creating a plateau from t0 to t1
        3. Shifting the right part (after t0) by (t1-t0)
        """
        if t1 < t0:
            raise ValueError("t1 must be greater than or equal to t0")
        
        if t1 == t0:
            return
            
        shift = t1 - t0
        new_segments = []
        
        shift = t1 - t0
        new_segments = []
        plateau_added = False
        
        # Process each segment
        for x_min, x_max, slope, intercept in self.segments:
            if x_max < t0:
                # Segment completely before t0 - keep unchanged
                new_segments.append((x_min, x_max, slope, intercept))
            elif x_min > t0:
                # Segment completely after t0 - shift right
                new_intercept = intercept - slope * shift
                new_segments.append((x_min + shift, x_max + shift, slope, new_intercept))
            else:
                # Segment contains or touches t0
                # Find value at t0
                val_at_t0 = slope * t0 + intercept
                
                # Add segment up to t0 if needed
                if x_min < t0:
                    new_segments.append((x_min, t0, slope, intercept))
                
                # Add plateau if we haven't yet
                if not plateau_added:
                    new_segments.append((t0, t1, 0, val_at_t0))
                    plateau_added = True
                
                # Add shifted remainder if there is any
                if x_max > t0:
                    new_segments.append((t1, x_max + shift, slope, val_at_t0 - slope * t1))
        
        self.segments = new_segments