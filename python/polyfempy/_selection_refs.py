import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SelectionRef:
    uuid: str
    namespace: str
    backend_id: int
    name: Optional[str] = None
    mesh_uuid: Optional[str] = None


class SelectionPool:
    def __init__(self, groups):
        self._groups = dict(groups)
        self._next_backend_id = {
            group: 1
            for group in set(self._groups.values())
        }

    def allocate(self, namespace, *, name=None, backend_id=None, mesh_uuid=None):
        if namespace not in self._groups:
            raise ValueError("unsupported selection namespace: %r" % namespace)

        id_group = self._groups[namespace]
        if backend_id is None:
            backend_id = self._next_backend_id[id_group]
        elif not isinstance(backend_id, int):
            raise TypeError("selection backend_id must be an int")
        elif backend_id < 1:
            raise ValueError("selection backend_id must be positive")

        self._next_backend_id[id_group] = max(
            self._next_backend_id[id_group],
            backend_id + 1,
        )

        return SelectionRef(
            uuid=str(uuid.uuid4()),
            namespace=namespace,
            backend_id=backend_id,
            name=name,
            mesh_uuid=mesh_uuid,
        )
