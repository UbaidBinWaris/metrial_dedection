module.exports = {
  MATERIALS: ["copper", "iron", "plastic", "aluminum", "glass", "wood"],
  PROFILE: {
    copper: {
      warmthMin: 0.08,
      saturationMin: 0.15,
      brightnessRange: [0.25, 0.8]
    },
    iron: {
      saturationMax: 0.16,
      brightnessRange: [0.18, 0.65],
      varianceMin: 0.07
    },
    plastic: {
      saturationMin: 0.2,
      varianceRange: [0.02, 0.2],
      brightnessRange: [0.2, 0.85]
    },
    aluminum: {
      saturationMax: 0.12,
      brightnessRange: [0.45, 0.95]
    },
    glass: {
      brightnessRange: [0.35, 0.95],
      saturationMax: 0.12,
      varianceMax: 0.16
    },
    wood: {
      warmthMin: 0.04,
      brightnessRange: [0.15, 0.7],
      varianceMin: 0.05
    }
  }
};
