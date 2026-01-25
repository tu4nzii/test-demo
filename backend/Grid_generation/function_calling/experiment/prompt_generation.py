def generate_prompt(
    item_name: str,
    prompt_type: str,
    x_ticks: list,
    y_ticks: list,
    pred_feedback: list = None,
    feedback_round: int = 0,
    current_round: int = 1
) -> str:
    x_tick_str = ", ".join(str(x) for x in x_ticks)
    y_tick_str = ", ".join(str(y) for y in y_ticks)

    if prompt_type == "baseline":
        return f'''
        You are given a chart image.
        Please extract the coordinates of the circle which represents [{item_name}]. Identify the graphical element that represents the target data item and extract its coordinates based on its visual center, not the location of any associated label or annotation.
        Only respond in this JSON format:
         {{"datapoints": [{{"{item_name}": [x, y]}}]}}
         '''

    elif prompt_type == "grid":
        return f'''
        You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        These grid lines divide the chart into rectangular cells aligned with axis ticks.
        The center of each visual mark (e.g., a circle) falls into a grid cell.
        The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
        Your task is to accurately extract the coordinates of the circle center representing [{item_name}] by:
        - Locating the center of the circle representing the data item;
        - Identifying its position between tick grid lines and using interpolation to estimate the (x, y) values of the circle center representing [{item_name}].

        Only respond in this JSON format:
        {{"datapoints": [{{"{item_name}": [x, y]}}]}}
        '''

    elif prompt_type == "feedback":
        base_prompt = f'''
        You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        These grid lines divide the chart into rectangular cells aligned with axis ticks.        
        The center of each visual mark (e.g., a circle) falls into a grid cell.
        The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
        Your task is to accurately extract the coordinates of the circle representing [{item_name}].
        '''

        if pred_feedback and isinstance(pred_feedback, list) and len(pred_feedback) >= 1:
            pred = pred_feedback[-1]
            base_prompt = f'''
            You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
            - X-axis ticks: [{x_tick_str}]
            - Y-axis ticks: [{y_tick_str}]
            These grid lines divide the chart into rectangular cells aligned with axis ticks.
            The center of each visual mark (e.g., a circle) falls into a grid cell.
            The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
            Your task is to accurately extract the coordinates of the circle representing [{item_name}].
            The predicted coordinates in the previous round is drawn as a red crosshair on the given chart.
            Please follow the steps below to refine your estimate for the coordinates of the circle representing [{item_name}]:
            Step 1: Find the circle that appears closest to the red crosshair and clarify its characteristics such as color, shape, and other visual features.
            Step 2: Compare the center of the circle with the red crosshair’s intersection, located at coordinates (x = {pred[0]:.2f}, y = {pred[1]:.2f}).
            Step 3: Identify the horizontal and vertical offset direction and distance between the red crosshair and the exact center.
            Step 4: Apply both the direction and magnitude of the offset to fine-tune your prediction.
            Note: the spacing between adjacent grid lines is 2.5 units in both directions.
            '''

        base_prompt += f'''
        Only respond in this JSON format:
        {{"datapoints": [{{"{item_name}": [x, y]}}]}}
        '''
        return base_prompt

    elif prompt_type == "feedback_cropped":
        return f'''
        You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        These grid lines divide the chart into rectangular cells aligned with axis ticks.        
        The center of each visual mark (e.g., a circle) falls into a grid cell.
        The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
        Your task is to accurately extract the coordinates of the circle center representing [{item_name}].

        Only respond in this JSON format:
        {{"datapoints": [{{"{item_name}": [x, y]}}]}}
        '''

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


if __name__ == "__main__":
    # ✅ 示例测试
    prompt = generate_prompt(
        item_name="C3",
        prompt_type="feedback",
        x_ticks=[0, 2.5, 5, 7.5, 10],
        y_ticks=[0, 2.5, 5, 7.5, 10],
        pred_feedback=[(3.2, 6.8)],
        current_round=2
    )
    print("=== 生成的 Prompt ===")
    print(prompt)
