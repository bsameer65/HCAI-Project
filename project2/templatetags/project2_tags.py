from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """
    Allow dict lookup with a dynamic key in templates.
    Usage: {{ my_dict|get_item:key_variable }}

    # HCAI extra: Django templates do not support dict[variable] natively,
    # so this tiny filter lets us look up per-class coefficients by class
    # name without hardcoding column names in the template.
    """
    coef_key = f"coef_{key}"
    return dictionary.get(coef_key, "")
