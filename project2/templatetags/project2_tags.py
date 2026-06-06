from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """
    Dynamic dict lookup using a coef_<key> pattern.
    Used for the LR coefficient table in train.html.
    Usage: {{ my_dict|get_item:key_variable }}

    # HCAI extra: Django templates do not support dict[variable] natively,
    # so this tiny filter lets us look up per-class coefficients by class
    # name without hardcoding column names in the template.
    """
    coef_key = f"coef_{key}"
    return dictionary.get(coef_key, "")


@register.filter
def get_dict_value(dictionary, key):
    """
    Plain dict[key] lookup for any key.
    Used in counterfactual.html to access feature values and change flags
    by dynamic feature name.
    Usage: {{ my_dict|get_dict_value:key_variable }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key, "")
    return ""
