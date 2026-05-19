Motion-Class Spectrum
8 motion classes from 23-D camera-subtracted optical-flow features. Each class shows 5 representative clips (closest to class centroid in 23-D space). Top row of each cell: original mp4. Bottom row: RAFT optical flow rendered via the Middlebury color wheel — hue = direction, saturation = magnitude.

🎬
Motion-Class Spectrum Gallery (8 classes × 5 clips + RAFT color-wheel viz)
pending: src/utils/motion_spectrum_gallery.py (planCODE_html.md Stage G — ~250 LoC). Inputs: data/eval_10k_local/motion_features.npy (9297, 23) + outputs/full/probe_action/action_labels.json (8 surviving classes). Outputs: 40 clip mp4s + 40 RAFT flow mp4s (~30 min GPU).

Within each row, all 5 clips will share a CONSISTENT color signature in the RAFT viz column — e.g., fast__rightward tints red, still__downward stays light with faint cyan. That visual constancy is the visceral proof that the 23-D feature actually captures direction + magnitude, even when the scenes differ (market vs drive vs walking).