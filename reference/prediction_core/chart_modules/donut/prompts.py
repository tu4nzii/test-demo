"""Prompt construction for donut-chart prediction flows."""

from __future__ import annotations

from typing import Dict, Optional, Union

def generate_prompt(
        item_name: str,
        prompt_type: str,
        prev_angle: Optional[Union[float, Dict[str, float]]] = None,
        drawn_angles: Optional[list[int]] = None,
        angle_order_hint: Optional[str] = None  # ✅ 新增参数
) -> str:
    # —— baseline ——
    if prompt_type == "baseline":
        prompt = f"""
        You are analyzing a pie chart. It shows data proportions using circular sectors, which divide the circle into slices. The size of each sector, represented by its central angle and area, corresponds to its proportion of the whole.
        Your task is to estimate the **percentage** value for the donut chart sector labeled "{item_name}".
        Output *only*:
        {{"datapoints": [{{"{item_name}": percentage}}]}}
        """
        return prompt

    # —— amplifier ——
    elif prompt_type == "amplifier":
        # 上一轮预测说明
        if isinstance(prev_angle, dict) and "start_angle" in prev_angle and "end_angle" in prev_angle:
            try:
                sa = float(prev_angle["start_angle"])
                ea = float(prev_angle["end_angle"])
                prev_str = (
                    f'The previous prediction for "{item_name}" defined a clockwise sector from the start angle to the end angle, that is'
                    f'**start: {sa:.1f}°**, **end: {ea:.1f}°**, the entire arc is considered the candidate range for this item.'
                )
            except Exception:
                prev_str = ""
        else:
            prev_str = ""

        # 注入网格线信息
        if drawn_angles:
            ticks_str = ", ".join(f"{a}°" for a in drawn_angles)
            grid_str = f"To support angle estimation, the visible radial tick marks in the image are drawn clockwise at: {ticks_str}."
        else:
            grid_str = ""

        # ✅ angle_order_hint 注入
        if angle_order_hint:
            order_hint_str = f"⚠️⚠️Note: {angle_order_hint}"
        else:
            order_hint_str = ""

        return f"""
        You are analyzing a **cropped sector** of "{item_name}" in the given donut chart.        
        {grid_str}        
        {prev_str}        
        Your task is to refine the estimation of the start and end angles (in degrees) of the sector labeled "{item_name}", such that the clockwise sector between them exactly corresponds to this labeled sector.          
        ️ There exist one principle to determine the correct start and end angle: the clockwise sector from the start angle to the end angle must contain only the color of the sector labeled "{item_name}".
        ⚠️ ⚠️ Note: **Enforce start_angle < end_angle.** ⚠️ ⚠️
        Instructions: 
        -Locate the sector labeled "{item_name}".   
        -Identify the start angle (x) of "{item_name}" — the angular position, measured clockwise.        
        -Identify the end angle (y) of "{item_name}" — the angular position, measured clockwise.
        Note: **The start_angle must smaller than the end_angle.**        

        Output *only* this JSON:
        {{"datapoints": [{{"{item_name}": {{"start_angle": start_angle, "end_angle": end_angle}}}}]}}
        """

    elif prompt_type == "feedback":
        if isinstance(prev_angle, dict) and "start_angle" in prev_angle and "end_angle" in prev_angle:
            prev_str = (
                f"The model previously predicted **start: {prev_angle['start_angle']:.1f}°**, "
                f"**end: {prev_angle['end_angle']:.1f}°**"
            )
        else:
            prev_str = ""

        prompt = f"""
                    You are analyzing a donut chart. It shows data proportions using ring-shaped sectors with a hollow center.                                
                    To support estimation, this chart includes **radial reference lines every 15°** that start from the right, dividing the full circle into 24 equal wedges.
                    There exist one principle to determine the correct start and end angle: the clockwise red arc between the two cross markers exactly matches the boundaries of the sector labeled "{item_name}".
                    ⚠️ ⚠️ Note: **Enforce start_angle < end_angle.** ⚠️ ⚠️
                    {prev_str} for the segment that represents "{item_name}", with the red visual feedback marks the predicted sector boundaries. 
                    A cross marker is drawn at the start and end angles, and a red arc connects them clockwise to indicate the predicted sector range.                   

                    Your task is to **refine both the start angle and the end angle** (in degrees) of the sector labeled **"{item_name}"** by:                    
                    1. You should first check if the red arc from the last prediction aligns with the true sector boundaries. 
                    - If it does, keep the current order of the start and end angles. If the red arc instead corresponds to the complementary region of the true sector, this means the start and end angles were reversed, and you must swap their order.
                    2. Identify the color of the sector representing "{item_name}", most likely located within the clockwise arc defined by the previous prediction’s start and end angles.                     
                    3. Compare the red visual feedback lines from last prediction with the true boundaries of the sector:                       
                       - If the red lines align with the true boundaries, keep the predictions.
                       - If not, adjust the **start and/or end angle** by adding or subtracting a few degrees, to make the prediction align with the true boundaries.                       
                    Note: **The start_angle must smaller than the end_angle.** 
                    Output *only*:
                    {{"datapoints": [{{"{item_name}": {{"start_angle": <float>, "end_angle": <float>}}}}]}}
                """

    else:  # grid

        prompt = f"""
                You are analyzing a donut chart. It shows data proportions using ring-shaped sectors with a hollow center.                        
                Your task is to estimate the start and end angles (in degrees) of the sector labeled "{item_name}", such that the clockwise sector between them exactly corresponds to this labeled sector.
                To support angle estimation, the chart includes angular reference lines every 15°, dividing the full circle into 24 equal wedges.
                These lines start from the right (0°) and proceed clockwise as follows:
                ['0°', 15°, 30°, 45°, 60°, 75°, 90°, 105°, 120°, 135°, 150°, 165°, 180°, 195°, 210°, 225°, 240°, 255°, 270°, 285°, 300°, 315°, 330°，345°].              
                There exist one principle to determine the correct start and end angle: the clockwise red arc between the two cross markers exactly matches the boundaries of the sector labeled "{item_name}".
                ⚠️ ⚠️ Note: **Enforce start_angle < end_angle.** ⚠️ ⚠️
                Instructions for accurately estimating the start and end angles (in degrees) of the sector labeled "{item_name}":
                Locate the sector labeled "{item_name}" on the outer ring.        
                Identify the start angle (x) — the angular position, measured clockwise, where the colored sector first begins. 
                Identify the end angle (y) — the angular position, measured clockwise, where the same sector finishes.         
                Use the reference lines to estimate each boundary angle as accurately as possible:        
                First, find the two nearest reference lines bracketing the boundary.        
                Then interpolate the sector’s position between them to compute a precise angle. 
                Note: **The start_angle must smaller than the end_angle.**              
                

                Output *only*:
                {{"datapoints": [{{"{item_name}": {{"start_angle": <float>, "end_angle": <float>}}}}]}}
                """

    return prompt.strip()
