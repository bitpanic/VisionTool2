import os
import importlib
import inspect
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem
from PyQt5.QtCore import pyqtSignal
from plugins.base_plugin import BasePlugin, FilterPlugin, DetectorPlugin

class PluginManager(QWidget):
    plugin_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.plugin_classes = {}
        self.load_plugins()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create plugin list
        self.plugin_list = QListWidget()
        self.plugin_list.itemDoubleClicked.connect(self.on_plugin_selected)
        layout.addWidget(self.plugin_list)

    def load_plugins(self):
        """Load all available plugins"""
        plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
        
        # Load filter plugins
        filters_dir = os.path.join(plugins_dir, 'filters')
        if os.path.exists(filters_dir):
            self._load_plugins_from_directory(filters_dir, 'filters')

        # Load detector plugins
        detectors_dir = os.path.join(plugins_dir, 'detectors')
        if os.path.exists(detectors_dir):
            self._load_plugins_from_directory(detectors_dir, 'detectors')

    def _load_plugins_from_directory(self, directory, category):
        """Load plugins from a specific directory"""
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                try:
                    # Import the module
                    module = importlib.import_module(f'plugins.{category}.{module_name}')
                    
                    # Find all plugin classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BasePlugin) and 
                            obj != BasePlugin and 
                            obj != FilterPlugin and 
                            obj != DetectorPlugin):
                            
                            # Create a temporary instance to get name and description
                            temp_instance = obj()
                            plugin_name = temp_instance.get_name()
                            
                            # Store the class
                            self.plugin_classes[plugin_name] = obj
                            
                            # Add to list widget
                            item = QListWidgetItem(plugin_name)
                            item.setToolTip(temp_instance.get_description())
                            self.plugin_list.addItem(item)
                except Exception as e:
                    print(f"Error loading plugin {module_name}: {str(e)}")

    def on_plugin_selected(self, item):
        """Handle plugin selection"""
        plugin_name = item.text()
        if plugin_name in self.plugin_classes:
            # Create a new instance when selected
            plugin = self.plugin_classes[plugin_name]()
            self.plugin_selected.emit(plugin)

    def get_plugin(self, name):
        """Get a new instance of a plugin by name"""
        if name in self.plugin_classes:
            return self.plugin_classes[name]()
        return None

    def get_all_plugins(self):
        """Get all loaded plugin classes"""
        return self.plugin_classes.copy()

    def create_plugin(self, plugin_name):
        """Create a new instance of a plugin"""
        if plugin_name in self.plugin_classes:
            return self.plugin_classes[plugin_name]()
        return None 