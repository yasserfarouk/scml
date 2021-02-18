from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from negmas import NamedObject

# from negmas.visualizers import WorldVisualizer, AgentVisualizer, Widget
#
#
# class SCML2019AgentVisualizer(AgentVisualizer):
#     @classmethod
#     def widget_kind(cls, widget_name: str) -> str:
#         pass
#
#     @classmethod
#     def widget_names(cls) -> List[str]:
#         return ["inventory_wallet"] + super().widget_names()
#
#     @classmethod
#     def widget_params(cls, widget_name: str) -> Dict[str, Type]:
#         pass
#
#     def render_widget(
#         self, name: str, params: Dict[str, Any] = None
#     ) -> Optional[Widget]:
#         pass
#
#     @classmethod
#     def default_widget_name(cls) -> str:
#         return "inventory_wallet"
#
#
# class FactoryManagerVisualizer(SCML2019AgentVisualizer):
#     @classmethod
#     def widget_kind(cls, widget_name: str) -> str:
#         pass
#
#     @classmethod
#     def widget_names(cls) -> List[str]:
#         return ["factory_schedule"] + super().widget_names()
#
#     @classmethod
#     def widget_params(cls, widget_name: str) -> Dict[str, Type]:
#         pass
#
#     def render_widget(
#         self, name: str, params: Dict[str, Any] = None
#     ) -> Optional[Widget]:
#         pass
#
#     @classmethod
#     def default_widget_name(cls) -> str:
#         return "factory_schedule"
#
#
# class SCML2019WorldVisualizer(WorldVisualizer):
#     @classmethod
#     def widget_kind(cls, widget_name: str) -> str:
#         pass
#
#     @classmethod
#     def widget_names(cls) -> List[str]:
#         pass
#
#     @classmethod
#     def widget_params(cls, widget_name: str) -> Dict[str, Type]:
#         pass
#
#     def render_widget(
#         self, name: str, params: Dict[str, Any] = None
#     ) -> Optional[Widget]:
#         pass
#
#     @classmethod
#     def default_widget_name(cls) -> str:
#         pass
#
#     @property
#     def children(self) -> Dict[str, List[NamedObject]]:
#         pass
#
#     @classmethod
#     def children_categories(cls) -> List[str]:
#         return [
#             "factory_managers",
#             "miners",
#             "consumers",
#         ] + super().children_categories()
#
#
# class BankVisualizer(SCML2019AgentVisualizer):
#     pass
#
#
# class InsuranceCompanyVisualizer(SCML2019AgentVisualizer):
#     pass
