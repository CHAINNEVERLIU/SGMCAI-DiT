def check_keys_exist(dictionary, keys):
    """
    检查字典中是否存在给定的多个键，并在缺少任何一个键时引发KeyError

    参数:
    dictionary: dict
        目标字典
    keys: list
        要检查的键列表

    返回:
    None
    """
    for key in keys:
        if key not in dictionary:
            raise KeyError(f"Must have parameter '{key}' in the dictionary '{dictionary}'!")
        else:
            pass
