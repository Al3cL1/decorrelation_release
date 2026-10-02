from typing import List

MT_TASKS = {
    # composites6: LIBERO-10 tasks whose two sub-goals are atomics25 tasks in the same scene.
    # atomics25: the single-skill LIBERO-90 tasks of those scenes, minus two chained ones.
    'atomics25': [
        # LIVING_ROOM_SCENE1 (4)
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        # LIVING_ROOM_SCENE2 (5)
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        # LIVING_ROOM_SCENE5 (4)
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
        # LIVING_ROOM_SCENE6 (4)
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
        # KITCHEN_SCENE3 (3 of 4 - composite excluded)
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove",
        # KITCHEN_SCENE4 (5 of 6 - composite excluded)
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    ],

    # 6 LIBERO-10 composites, each two atomics25 tasks (its sub-tasks: 'decomp12', DECOMP_PARENT)
    'composites6': [
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    ],

    # the two sub-tasks of each composites6 task
    'decomp12': [
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "KITCHEN_SCENE3_turn_on_the_stove",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    ],
}

# sub-task -> its composites6 task
DECOMP_PARENT = {
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket":
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket":
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket":
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket":
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate":
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate":
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate":
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate":
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "KITCHEN_SCENE3_turn_on_the_stove":
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove":
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet":
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet":
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]

