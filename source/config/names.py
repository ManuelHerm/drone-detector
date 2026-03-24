"""This module stores the names for various entities."""

# pylint: disable=missing-class-docstring
# pylint: disable=too-few-public-methods
# Disabled because these classes are trivial
#     and just containers for specific names.


class DataOriginNames:
    cranfield_default: str = "Cranfield: Drones Only 40m"
    cranfield_distractors: str = "Cranfield: Generic Distractors 40m"
    drone_vs_bird: str = "Drone versus Bird"
    current_work: str = "Thesis"
    current_work_meadow: str = "Thesis Meadow Scene"
    current_work_city: str = "Thesis City Scene"
    current_work_lake: str = "Thesis Lake Scene"
    current_work_disorder: str = "Thesis Disorder Secene"
    current_work_lanterns: str = "Thesis Lantern Scene"
    current_work_lanterns_full: str = "Thesis Lantern 100"
    current_work_container: str = "Thesis Container Scene"
    current_work_container_full: str = "Thesis Container 100"
    current_work_disorder_full: str = "Thesis Disorder 100"
    current_work_pine_twigs: str = "Thesis Pine Twigs"
    current_work_normal_twigs: str = "Thesis Tree Twigs"
    current_work_parrot_one: str = "Thesis Parrot One"
    current_work_parrot_two: str = "Thesis Parrot Two"
    current_work_moon: str = "Thesis Moon"


class DatasetNames:
    cranfield_default: str = "Cranfield: Drones Only 40m"
    cranfield_distractors: str = "Cranfield: Generic Distractors 40m"
    cranfield_combined: str = (
        "Cranfield: Combined Simple Drones and Generic Distractors 40m"
    )

    cranfield_combined_section: str = "Cranfield: Combined Section"
    cranfield_combined_section_larger: str = "Cranfield: Combined Larger Section"
    cranfield_default_and_current_section: str = (
        "Complete Cranfield Default and Part of Campus, "
        "Forrest, Hill, and Drone School Mid"
    )

    drone_vs_bird: str = "Drone versus Bird"

    forrest_hut_phantom: str = "Forrest Hut with DJI Phantom"
    forrest_hut_phantom_section: str = "Part of Forrest Hut with Phantom Drone"
    forrest_hut_mini: str = "Forrest Hut with DJI Mini 3"
    forrest_hut_mini_section: str = "Part of Forrest Hut with Mini Drone"
    forrest_hut_inspire: str = "Forrest Hut with DJI Inspire"
    forrest_hut_inspire_section: str = "Part of Forrest Hut with Inspire Drone"
    forrest_hut_section: str = "Part of Forrest Hut With All Three Drones"

    campus: str = "Campus with All Three Drones"
    campus_section: str = "Part of Campus Dataset"

    hill: str = "Hill with all Three Drone"
    hill_section: str = "Part of Hill Dataset"

    drone_schools: str = "Drone Schools"
    drone_school_phantom: str = "Drone School with DJI Phantom"
    drone_school_mini: str = "Drone School with DJI Mini"
    drone_school_inspire: str = "Drone School with DJI Inspire"
    drone_school_mid: str = "Drone School at Medium Distance With All Three Drones"
    drone_school_mid_section: str = "Part of Drone School at Medium Distance"
    drone_school_far: str = "Drone School at Big Distance With All Three Drones"

    real_world_trees: str = "trees-background"
    real_world_sky: str = "sky-background"

    meadow: str = "Meadow"
    meadow_section: str = "Meadow Section"

    city: str = "City"
    city_section: str = "City Section"

    lake: str = "Lake"
    lake_section: str = "Lake Section"

    disorder_full: str = "Disorder - All images"
    disorder_section: str = "Disorder"

    lanterns_full: str = "Lanterns - All images"
    lanterns_section: str = "Lanterns"

    container_full: str = "Container - All images"
    container_section: str = "Container"

    twigs_pine: str = "Pine Twigs"
    twigs_tree: str = "Tree Twigs"

    parrot_one: str = "Parrot One"
    parrot_two: str = "Parrot Two"

    moon: str = "Moon"

    campus_forrest_hill_school_section: str = (
        "Part of Campus, Forrest, Hill, and Drone School Mid Dataset"
    )
    current_and_cranfield_default: str = "Current Work and Cranfield's Drones Only"

    campus_hill: str = "Campus"
    all_combined: str = "Current Work All Combined"

    hill_campus_forrest: str = "Campus, Hill, and Forrest"

    best_shot: str = (
        "Best attempt: Cranfield combined + "
        "Sections of Campus, Hill, Forrst, Lake, City, and Meadow"
    )
    next_best_shot: str = (
        "Next best attempt: Cranfield combined + "
        "Sections of Campus, Hill, Forrst, Lake, City, Meadow, and Disorder"
    )
    optimization_0: str = "Next best shot + Lanterns"
    optimization_1: str = (
        "As optimization_0 but with section of Cranfield combined dataset"
    )
    optimization_2: str = (
        "As optimization_1 but with larger section of Cranfield combined dataset"
    )

    thesis_new: str = "Cranfield Default + New Data per Thesis Description"

    optimization_3: str = "Cranfield Default + New Data + Parrot + Moon"


class DataCategoryNames:
    training: str = "training"
    validation: str = "validation"
    testing: str = "testing"


class DroneVsBirdVideos:
    video_names: tuple[str, ...] = (
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

    hard_videos_thesis = (
        "dji_mavick_hillside_off_focus",
        "gopro_007",
        "2019_10_16_C0003_3633_inspire",
        "2019_08_19_GP015869_1520_inspire",
        "parrot_clear_birds",
        "two_distant_phantom",
        "gopro_006",
        "dji_phantom_4_hillside_cross",
        "GOPR5848_002",
        "dji_phantom_4_long_takeoff",
        "distant_parrot_with_birds",
        "gopro_002",
        "GOPR5843_002",
        "dji_phantom_4_swarm_noon",
        "00_01_52_to_00_01_58",
        "parrot_disco_distant_cross",
        "GOPR5846_002",
        "swarm_dji_phantom",
        "GOPR5844_004",
        "dji_mavick_distant_hillside",
        "gopro_005",
        "parrot_disco_distant_cross_3",
        "parrot_clear_birds_med_range",
        "2019_09_02_GOPR5871_1058_solo",
        "off_focus_parrot_birds",
        "gopro_004",
        "2019_08_19_GOPR5869_1530_phantom",
        "gopro_000",
        "gopro_001",
        "GOPR5842_005",
        "GOPR5844_002",
        "gopro_008",
        "parrot_disco_long_session",
    )

    easy_videos_thesis = (
        "dji_matrice_210_sky",
        "2019_10_16_C0003_5043_mavic",
        "parot_disco_takeoff",
        "00_06_10_to_00_06_27",
    )

    better_1st_data_addition = (
        "2019_09_02_GOPR5871_1058_solo",
        "gopro_003",
        "2019_08_19_GOPR5869_1530_phantom",
        "GOPR5845_001",
        "gopro_002",
        "GOPR5847_004",
        "2019_10_16_C0003_4613_mavic",
        "GOPR5844_004",
    )

    worse_1st_data_addition = (
        "two_uavs_plus_airplane",
        "off_focus_parrot_birds",
        "GOPR5848_002",
        "GOPR5846_002",
        "parrot_clear_birds_med_range",
    )

    validation_videos = (
        "2019_09_02_GOPR5871_1058_solo",
        "gopro_001",
        "gopro_004",
        "00_02_45_to_00_03_10_cut",
    )

    extended_validation = (
        "GOPR5845_001",
        "parrot_disco_distant_cross",
        "parrot_disco_midrange_cross",
        "GOPR5842_005",
        "GOPR5844_002",
        "GOPR5847_003",
        "two_parrot_disco_1",
        "two_distant_phantom",
        "GOPR5845_004",
        "GOPR5846_005",
        "dji_pantom_landing_custom_fixed_takeoff",
        "2019_08_19_GOPR5869_1530_phantom",
        "fixed_wing_over_hill_1",
        "parrot_disco_distant_cross_3",
        "2019_08_19_GP015869_1520_inspire",
        "GOPR5842_007",
        "dji_phantom_4_swarm_noon",
        "custom_fixed_wing_1",
        "gopro_002",
        "gopro_001",
        "fixed_wing_over_hill_2",
        "matrice_600_2",
        "GOPR5843_002",
        "two_uavs_plus_airplane",
        "00_02_45_to_00_03_10_cut",
        "GOPR5848_002",
        "custom_fixed_wing_2",
        "2019_09_02_GOPR5871_1058_solo",
        "GOPR5842_002",
        "matrice_600_3",
        "GOPR5844_004",
        "gopro_000",
        "dji_matrice_210_sky",
        "swarm_dji_phantom4_2",
        "swarm_dji_phantom",
        "00_01_52_to_00_01_58",
        "gopro_003",
        "GOPR5843_005",
    )
