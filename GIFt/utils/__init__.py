from typing import Sequence

def default(object,default_value):
    """
    Returns the default value if the object is None.

    Args:
        object (Any): The object to check.
        default_value (Any): The default value to return if the object is None.

    Returns:
        Any: The object if it is not None, otherwise the default value.
    """
    return object if object is not None else default_value

def get_class_name(obj):
    return obj.__class__.__name__

def take_intersection(sets:Sequence[Sequence]):
    initial_set=sets[0]
    intersection=[]
    for element in initial_set:
        not_in_all=False    
        for set_i in sets[1:]:
            if element not in set_i:
                not_in_all=True
                break
        if not not_in_all:
            intersection.append(element)
    return intersection