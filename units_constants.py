class UnitsAndConstants:
    def __init__(self):
        self.nm = 1e-9        # Nanometers to meters
        self.mm = 1e-3        # Millimeters to meters
        self.cm = 1e-2        # Centimeters to meters
        self.eV = 1.6e-19     # Electronvolt to joules

units = UnitsAndConstants()

# Access the constants
nm = units.nm
mm = units.mm
cm = units.cm
eV = units.eV
