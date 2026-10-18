"""Stand-in for `jaxtyping`: `scene_language/engine/utils/type_utils.py` only uses `Float[np.ndarray, "n"]`
as an annotation. Each name subscripts to the array type it wraps."""


class _Annot:
    def __init__(self, name):
        self.__name__ = name

    def __getitem__(self, item):
        return item[0] if isinstance(item, tuple) else item


Float = _Annot("Float")
Int = _Annot("Int")
Bool = _Annot("Bool")
Shaped = _Annot("Shaped")
Array = object
