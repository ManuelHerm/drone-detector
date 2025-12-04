"""This module stores the names for various entities."""

# pylint: disable=missing-class-docstring
# pylint: disable=too-few-public-methods
# Disabled because these classes are trivial
#     and just containers for specific names.


class DataOriginNames:
    cranfield_default: str = "Cranfield: Drones Only 40m"
    cranfield_distractors: str = "Cranfield: Generic Distractors 40m"
    drone_vs_bird: str = "Drone versus Bird"


class DatasetNames:
    cranfield_default: str = "Cranfield: Drones Only 40m"
    cranfield_distractors: str = "Cranfield: Generic Distractors 40m"
    drone_vs_bird: str = "Drone versus Bird"
    cranfield_combined: str = (
        "Cranfield: Combined Simple Drones and Generic Distractors 40m"
    )


class DataCategoryNames:
    training: str = "training"
    validation: str = "validation"
    testing: str = "testing"


class DroneVsBirdVideos:
    video_names = (
        "00_01_52_to_00_01_58",
        "00_02_45_to_00_03_10_cut",
        "00_06_10_to_00_06_27",
        "00_09_30_to_00_10_09",
        "00_10_09_to_00_10_40",
        "2019_08_19_C0001_5319_phantom",
        "2019_08_19_GOPR5869_1530_phantom",
        "2019_08_19_GP015869_1520_inspire",
        "2019_09_02_C0002_2527_inspire",
        "2019_09_02_C0002_3700_mavic",
        "2019_09_02_GOPR5871_1058_solo",
        "2019_10_16_C0003_1700_matrice",
        "2019_10_16_C0003_3633_inspire",
        "2019_10_16_C0003_4613_mavic",
        "2019_10_16_C0003_5043_mavic",
        "2019_11_14_C0001_3922_matrice",
        "GOPR5842_002",
        "GOPR5842_005",
        "GOPR5842_007",
        "GOPR5843_002",
        "GOPR5843_005",
        "GOPR5844_002",
        "GOPR5844_004",
        "GOPR5845_001",
        "GOPR5845_004",
        "GOPR5846_002",
        "GOPR5846_005",
        "GOPR5847_003",
        "GOPR5847_004",
        "GOPR5848_002",
        "GOPR5848_004",
        "custom_fixed_wing_1",
        "custom_fixed_wing_2",
        "distant_parrot_2",
        "distant_parrot_with_birds",
        "dji_matrice_210_hillside",
        "dji_matrice_210_mountain",
        "dji_matrice_210_off_focus",
        "dji_matrice_210_sky",
        "dji_mavick_close_buildings",
        "dji_mavick_distant_hillside",
        "dji_mavick_hillside_off_focus",
        "dji_mavick_mountain",
        "dji_mavick_mountain_cruise",
        "dji_pantom_landing_custom_fixed_takeoff",
        "dji_phantom_4_hillside_cross",
        "dji_phantom_4_long_takeoff",
        "dji_phantom_4_mountain_hover",
        "dji_phantom_4_swarm_noon",
        "dji_phantom_mountain_cross",
        "fixed_wing_over_hill_1",
        "fixed_wing_over_hill_2",
        "gopro_000",
        "gopro_001",
        "gopro_002",
        "gopro_003",
        "gopro_004",
        "gopro_005",
        "gopro_006",
        "gopro_007",
        "gopro_008",
        "matrice_600_2",
        "matrice_600_3",
        "off_focus_parrot_birds",
        "parot_disco_takeoff",
        "parrot_clear_birds",
        "parrot_clear_birds_med_range",
        "parrot_disco_distant_cross",
        "parrot_disco_distant_cross_3",
        "parrot_disco_long_session",
        "parrot_disco_midrange_cross",
        "parrot_disco_zoomin_zoomout",
        "swarm_dji_phantom",
        "swarm_dji_phantom4_2",
        "two_distant_phantom",
        "two_parrot_disco_1",
        "two_uavs_plus_airplane",
    )

    hard_videos = (  # Perform worse than average on model state 80 / or after
        # about a million training samples from the combined
        # Cranfield dataset.
        "2019_09_02_GOPR5871_1058_solo",
        "gopro_007",
        "dji_mavick_hillside_off_focus",
        "2019_08_19_GP015869_1520_inspire",
        "swarm_dji_phantom",
        "dji_phantom_4_hillside_cross",
        "GOPR5848_002",
        "swarm_dji_phantom4_2",
        "2019_10_16_C0003_3633_inspire",
        "dji_phantom_4_swarm_noon",
        "gopro_006",
        "gopro_000",
        "dji_phantom_4_long_takeoff",
        "two_distant_phantom",
        "GOPR5846_002",
        "parrot_disco_distant_cross",
        "dji_pantom_landing_custom_fixed_takeoff",
        "custom_fixed_wing_1",
        "parrot_disco_distant_cross_3",
        "GOPR5846_005",
        "GOPR5843_002",
        "2019_10_16_C0003_4613_mavic",
        "gopro_002",
        "GOPR5844_002",
        "distant_parrot_with_birds",
        "gopro_005",
        "parrot_clear_birds",
        "parrot_clear_birds_med_range",
        "2019_08_19_GOPR5869_1530_phantom",
        "gopro_008",
        "GOPR5844_004",
        "dji_mavick_distant_hillside",
        "off_focus_parrot_birds",
        "parrot_disco_long_session",
        "2019_08_19_C0001_5319_phantom",
        "custom_fixed_wing_2",
        "gopro_001",
        "two_parrot_disco_1",
    )
