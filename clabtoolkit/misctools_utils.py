# misctools_utils.py
import numpy as np
from collections import defaultdict
from typing import Any, Optional, Dict, List
import sys

# Try to import required modules at module level
try:
    from clabtoolkit import cltcolors

    HAS_CLTCOLORS = True
except ImportError:
    HAS_CLTCOLORS = False

try:
    from IPython.display import display, HTML
    from IPython import get_ipython

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class ExplorerDict(dict):
    """
    Enhanced dictionary with built-in exploration and visualization methods.

    Works like a regular dict but adds methods for tree visualization,
    summaries, searching, and safe updating.

    Examples
    --------
    >>> data = ExplorerDict({
    ...     'subject': 'sub-001',
    ...     'metadata': {'age': 25, 'sessions': ['ses-01', 'ses-02']}
    ... })
    >>> data.tree()           # Print tree visualization
    >>> data.summary()        # Get statistics
    >>> data.search('age')    # Find keys
    """

    def __init__(
        self, *args, name: str = "root", force_mode: Optional[str] = None, **kwargs
    ):
        """
        Initialize ExplorerDict.

        Parameters
        ----------
        *args, **kwargs
            Same as dict initialization
        name : str
            Name for the root object (for display)
        force_mode : str, optional
            Force display mode: 'notebook', 'terminal', or None (auto-detect)
        """
        super().__init__(*args, **kwargs)
        self._name = name

        if force_mode is not None:
            self._notebook_mode = force_mode == "notebook"
        else:
            self._notebook_mode = self._is_notebook()

        if self._notebook_mode and not HAS_IPYTHON:
            self._notebook_mode = False

    @staticmethod
    def _is_notebook():
        """Check if running in Jupyter notebook."""
        if not HAS_IPYTHON:
            return False
        try:
            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except (NameError, AttributeError):
            return False

    def tree(
        self,
        max_depth: Optional[int] = None,
        max_items: int = 10,
        max_str_len: int = 50,
        show_types: bool = True,
        show_shapes: bool = True,
    ) -> None:
        """
        Print tree visualization of the dictionary structure.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to display
        max_items : int
            Maximum items to show per level
        max_str_len : int
            Maximum string length before truncation
        show_types : bool
            Show type information
        show_shapes : bool
            Show array shapes

        Examples
        --------
        >>> data = ExplorerDict({'a': [1, 2, 3], 'b': {'c': 4}})
        >>> data.tree()
        🌳 root
        ├── a: [1, 2, 3] (list, len=3)
        └── b: (dict, len=1)
            └── c: 4 (int)
        """
        if self._notebook_mode:
            self._tree_html(max_depth, max_items, max_str_len, show_types, show_shapes)
        else:
            self._tree_terminal(
                max_depth, max_items, max_str_len, show_types, show_shapes
            )

    def _tree_terminal(
        self, max_depth, max_items, max_str_len, show_types, show_shapes
    ):
        """Terminal output with ANSI colors."""
        if HAS_CLTCOLORS:
            print(
                f"{cltcolors.bcolors.HEADER}{cltcolors.bcolors.BOLD}🌳 {self._name}{cltcolors.bcolors.ENDC}"
            )
            print(f"{cltcolors.bcolors.OKGRAY}{'─' * 60}{cltcolors.bcolors.ENDC}")
        else:
            print(f"🌳 {self._name}")
            print("─" * 60)

        self._print_tree_terminal(
            self,
            "",
            True,
            max_depth,
            0,
            max_items,
            max_str_len,
            show_types,
            show_shapes,
        )
        print()

    def _print_tree_terminal(
        self,
        obj,
        indent,
        is_last,
        max_depth,
        depth,
        max_items,
        max_str_len,
        show_types,
        show_shapes,
    ):
        """Recursive tree printer for terminal."""
        if max_depth is not None and depth >= max_depth:
            connector = "└── " if is_last else "├── "
            if HAS_CLTCOLORS:
                print(
                    f"{indent}{cltcolors.bcolors.OKGRAY}{connector}<max depth>{cltcolors.bcolors.ENDC}"
                )
            else:
                print(f"{indent}{connector}<max depth>")
            return

        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if isinstance(obj, dict):
            items = list(obj.items())[:max_items]
            truncated = len(obj) > max_items

            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1) and not truncated
                type_info = self._get_type_info(value, show_types, show_shapes)

                if isinstance(value, (dict, list, tuple)):
                    if HAS_CLTCOLORS:
                        print(
                            f"{indent}{cltcolors.bcolors.OKWHITE}{connector}"
                            f"{cltcolors.bcolors.DARKCYAN}{key}{cltcolors.bcolors.ENDC}: "
                            f"{cltcolors.bcolors.OKGRAY}{type_info}{cltcolors.bcolors.ENDC}"
                        )
                    else:
                        print(f"{indent}{connector}{key}: {type_info}")

                    self._print_tree_terminal(
                        value,
                        indent + extension,
                        is_last_item,
                        max_depth,
                        depth + 1,
                        max_items,
                        max_str_len,
                        show_types,
                        show_shapes,
                    )
                else:
                    val_str = self._format_value(value, max_str_len)
                    if HAS_CLTCOLORS:
                        print(
                            f"{indent}{cltcolors.bcolors.OKWHITE}{connector}"
                            f"{cltcolors.bcolors.DARKCYAN}{key}{cltcolors.bcolors.ENDC}: "
                            f"{cltcolors.bcolors.OKPURPLE}{val_str}{cltcolors.bcolors.ENDC} "
                            f"{cltcolors.bcolors.OKGRAY}{type_info}{cltcolors.bcolors.ENDC}"
                        )
                    else:
                        print(f"{indent}{connector}{key}: {val_str} {type_info}")

            if truncated:
                if HAS_CLTCOLORS:
                    print(
                        f"{indent}{cltcolors.bcolors.OKGRAY}└── "
                        f"... [{len(obj) - max_items} more items]{cltcolors.bcolors.ENDC}"
                    )
                else:
                    print(f"{indent}└── ... [{len(obj) - max_items} more items]")

        elif isinstance(obj, (list, tuple)):
            items = obj[:max_items]
            truncated = len(obj) > max_items

            for i, item in enumerate(items):
                is_last_item = (i == len(items) - 1) and not truncated
                type_info = self._get_type_info(item, show_types, show_shapes)

                if isinstance(item, (dict, list, tuple)):
                    if HAS_CLTCOLORS:
                        print(
                            f"{indent}{cltcolors.bcolors.OKWHITE}{connector}"
                            f"{cltcolors.bcolors.OKYELLOW}[{i}]{cltcolors.bcolors.ENDC}: "
                            f"{cltcolors.bcolors.OKGRAY}{type_info}{cltcolors.bcolors.ENDC}"
                        )
                    else:
                        print(f"{indent}{connector}[{i}]: {type_info}")

                    self._print_tree_terminal(
                        item,
                        indent + extension,
                        is_last_item,
                        max_depth,
                        depth + 1,
                        max_items,
                        max_str_len,
                        show_types,
                        show_shapes,
                    )
                else:
                    val_str = self._format_value(item, max_str_len)
                    if HAS_CLTCOLORS:
                        print(
                            f"{indent}{cltcolors.bcolors.OKWHITE}{connector}"
                            f"{cltcolors.bcolors.OKYELLOW}[{i}]{cltcolors.bcolors.ENDC}: "
                            f"{cltcolors.bcolors.OKPURPLE}{val_str}{cltcolors.bcolors.ENDC} "
                            f"{cltcolors.bcolors.OKGRAY}{type_info}{cltcolors.bcolors.ENDC}"
                        )
                    else:
                        print(f"{indent}{connector}[{i}]: {val_str} {type_info}")

            if truncated:
                if HAS_CLTCOLORS:
                    print(
                        f"{indent}{cltcolors.bcolors.OKGRAY}└── "
                        f"... [{len(obj) - max_items} more items]{cltcolors.bcolors.ENDC}"
                    )
                else:
                    print(f"{indent}└── ... [{len(obj) - max_items} more items]")

    def _tree_html(self, max_depth, max_items, max_str_len, show_types, show_shapes):
        """Notebook output with HTML formatting - Dark theme."""
        if not HAS_IPYTHON:
            # Fallback to terminal
            self._tree_terminal(
                max_depth, max_items, max_str_len, show_types, show_shapes
            )
            return

        html = f"""
        <div style="font-family: 'Courier New', monospace; line-height: 1.4; 
                    background: #1e1e1e; color: #d4d4d4; padding: 15px; 
                    border-radius: 5px; border: 1px solid #3c3c3c;">
            <h3 style="color: #c586c0; margin: 0 0 10px 0; font-weight: normal;">🌳 {self._name}</h3>
            <div style="border-top: 2px solid #3c3c3c; padding-top: 10px;">
        """

        html += self._build_tree_html(
            self,
            "",
            True,
            max_depth,
            0,
            max_items,
            max_str_len,
            show_types,
            show_shapes,
        )

        html += """
            </div>
        </div>
        """
        display(HTML(html))

    def _build_tree_html(
        self,
        obj,
        indent,
        is_last,
        max_depth,
        depth,
        max_items,
        max_str_len,
        show_types,
        show_shapes,
    ):
        """Recursively build HTML tree - Dark theme."""
        html = ""

        if max_depth is not None and depth >= max_depth:
            connector = "└── " if is_last else "├── "
            html += f'{indent}<span style="color: #6a6a6a;">{connector}&lt;max depth&gt;</span><br>'
            return html

        connector = "└── " if is_last else "├── "
        extension_char = (
            "&nbsp;&nbsp;&nbsp;&nbsp;" if is_last else "│&nbsp;&nbsp;&nbsp;"
        )

        if isinstance(obj, dict):
            items = list(obj.items())[:max_items]
            truncated = len(obj) > max_items

            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1) and not truncated
                type_info = self._get_type_info(value, show_types, show_shapes)

                if isinstance(value, (dict, list, tuple)):
                    html += (
                        f'{indent}<span style="color: #858585;">{connector}</span>'
                        f'<span style="color: #4ec9b0; font-weight: bold;">{key}</span>: '
                        f'<span style="color: #808080;">{type_info}</span><br>'
                    )
                    html += self._build_tree_html(
                        value,
                        indent + extension_char,
                        is_last_item,
                        max_depth,
                        depth + 1,
                        max_items,
                        max_str_len,
                        show_types,
                        show_shapes,
                    )
                else:
                    val_str = self._format_value(value, max_str_len)
                    html += (
                        f'{indent}<span style="color: #858585;">{connector}</span>'
                        f'<span style="color: #4ec9b0; font-weight: bold;">{key}</span>: '
                        f'<span style="color: #ce9178;">{val_str}</span> '
                        f'<span style="color: #808080;">{type_info}</span><br>'
                    )

            if truncated:
                html += (
                    f'{indent}<span style="color: #6a6a6a;">└── '
                    f"... [{len(obj) - max_items} more items]</span><br>"
                )

        elif isinstance(obj, (list, tuple)):
            items = obj[:max_items]
            truncated = len(obj) > max_items

            for i, item in enumerate(items):
                is_last_item = (i == len(items) - 1) and not truncated
                type_info = self._get_type_info(item, show_types, show_shapes)

                if isinstance(item, (dict, list, tuple)):
                    html += (
                        f'{indent}<span style="color: #858585;">{connector}</span>'
                        f'<span style="color: #dcdcaa;">[{i}]</span>: '
                        f'<span style="color: #808080;">{type_info}</span><br>'
                    )
                    html += self._build_tree_html(
                        item,
                        indent + extension_char,
                        is_last_item,
                        max_depth,
                        depth + 1,
                        max_items,
                        max_str_len,
                        show_types,
                        show_shapes,
                    )
                else:
                    val_str = self._format_value(item, max_str_len)
                    html += (
                        f'{indent}<span style="color: #858585;">{connector}</span>'
                        f'<span style="color: #dcdcaa;">[{i}]</span>: '
                        f'<span style="color: #ce9178;">{val_str}</span> '
                        f'<span style="color: #808080;">{type_info}</span><br>'
                    )

            if truncated:
                html += (
                    f'{indent}<span style="color: #6a6a6a;">└── '
                    f"... [{len(obj) - max_items} more items]</span><br>"
                )

        return html

    def _get_type_info(self, obj, show_types, show_shapes):
        """Get formatted type and shape information."""
        if not show_types and not show_shapes:
            return ""

        info_parts = []

        if isinstance(obj, np.ndarray):
            if show_shapes:
                info_parts.append(f"shape={obj.shape}")
            if show_types:
                info_parts.append(f"dtype={obj.dtype}")
        elif isinstance(obj, (list, tuple)):
            if show_types:
                info_parts.append(f"{type(obj).__name__}")
            info_parts.append(f"len={len(obj)}")
        elif isinstance(obj, dict):
            if show_types:
                info_parts.append("dict")
            info_parts.append(f"len={len(obj)}")
        elif show_types:
            info_parts.append(f"{type(obj).__name__}")

        return f"({', '.join(info_parts)})" if info_parts else ""

    def _format_value(self, value, max_len):
        """Format a value for display."""
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return "empty array"
            elif value.size <= 5:
                return str(value)
            else:
                return f"array([{value.flat[0]}, ..., {value.flat[-1]}])"

        val_str = str(value)
        if len(val_str) > max_len:
            return val_str[:max_len] + "..."
        return val_str

    def summary(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Parameters
        ----------
        verbose : bool
            Show detailed information

        Returns
        -------
        dict
            Statistics about the dictionary structure

        Examples
        --------
        >>> data = ExplorerDict({'a': [1, 2], 'b': {'c': 3}})
        >>> stats = data.summary()
        📊 Dictionary Summary: root
        ...
        """
        if self._notebook_mode:
            return self._summary_html(verbose)
        else:
            return self._summary_terminal(verbose)

    def _summary_terminal(self, verbose):
        """Terminal summary with ANSI colors."""
        stats = self._collect_stats(self)

        if HAS_CLTCOLORS:
            print(
                f"{cltcolors.bcolors.HEADER}{cltcolors.bcolors.BOLD}{'='*60}{cltcolors.bcolors.ENDC}"
            )
            print(
                f"{cltcolors.bcolors.HEADER}{cltcolors.bcolors.BOLD}📊 Dictionary Summary: {self._name}{cltcolors.bcolors.ENDC}"
            )
            print(
                f"{cltcolors.bcolors.HEADER}{cltcolors.bcolors.BOLD}{'='*60}{cltcolors.bcolors.ENDC}"
            )
            print(
                f"{cltcolors.bcolors.OKBLUE}Root type:{cltcolors.bcolors.ENDC} {type(self).__name__}"
            )
            print(
                f"{cltcolors.bcolors.OKBLUE}Top-level keys:{cltcolors.bcolors.ENDC} {len(self)}"
            )

            print(
                f"\n{cltcolors.bcolors.OKGREEN}{cltcolors.bcolors.BOLD}Structure:{cltcolors.bcolors.ENDC}"
            )
            print(
                f"  {cltcolors.bcolors.DARKCYAN}Total dictionaries:{cltcolors.bcolors.ENDC} {stats['n_dicts']}"
            )
            print(
                f"  {cltcolors.bcolors.DARKCYAN}Total lists/tuples:{cltcolors.bcolors.ENDC} {stats['n_lists']}"
            )
            print(
                f"  {cltcolors.bcolors.DARKCYAN}Total keys:{cltcolors.bcolors.ENDC} {stats['n_keys']}"
            )
            print(
                f"  {cltcolors.bcolors.DARKCYAN}Maximum depth:{cltcolors.bcolors.ENDC} {stats['max_depth']}"
            )

            print(
                f"\n{cltcolors.bcolors.OKGREEN}{cltcolors.bcolors.BOLD}Data types found:{cltcolors.bcolors.ENDC}"
            )
            for dtype, count in sorted(stats["types"].items(), key=lambda x: -x[1])[
                :10
            ]:
                print(
                    f"  {cltcolors.bcolors.OKPURPLE}{dtype}:{cltcolors.bcolors.ENDC} {count}"
                )

            if stats["arrays"]:
                print(
                    f"\n{cltcolors.bcolors.OKGREEN}{cltcolors.bcolors.BOLD}NumPy arrays:{cltcolors.bcolors.ENDC}"
                )
                print(
                    f"  {cltcolors.bcolors.DARKCYAN}Count:{cltcolors.bcolors.ENDC} {len(stats['arrays'])}"
                )
                print(
                    f"  {cltcolors.bcolors.DARKCYAN}Total elements:{cltcolors.bcolors.ENDC} {sum(a['size'] for a in stats['arrays'])}"
                )
                if verbose:
                    shapes = [a["shape"] for a in stats["arrays"][:5]]
                    shapes_str = ", ".join(str(s) for s in shapes)
                    if len(stats["arrays"]) > 5:
                        shapes_str += ", ..."
                    print(
                        f"  {cltcolors.bcolors.DARKCYAN}Shapes:{cltcolors.bcolors.ENDC} {shapes_str}"
                    )

            print(
                f"\n{cltcolors.bcolors.OKGREEN}{cltcolors.bcolors.BOLD}Memory:{cltcolors.bcolors.ENDC}"
            )
            print(
                f"  {cltcolors.bcolors.DARKCYAN}Estimated size:{cltcolors.bcolors.ENDC} {self._format_bytes(stats['memory'])}"
            )
            print(
                f"{cltcolors.bcolors.HEADER}{cltcolors.bcolors.BOLD}{'='*60}{cltcolors.bcolors.ENDC}\n"
            )
        else:
            print("=" * 60)
            print(f"📊 Dictionary Summary: {self._name}")
            print("=" * 60)
            print(f"Root type: {type(self).__name__}")
            print(f"Top-level keys: {len(self)}")
            print(f"\nStructure:")
            print(f"  Total dictionaries: {stats['n_dicts']}")
            print(f"  Total lists/tuples: {stats['n_lists']}")
            print(f"  Total keys: {stats['n_keys']}")
            print(f"  Maximum depth: {stats['max_depth']}")
            print(f"\nData types found:")
            for dtype, count in sorted(stats["types"].items(), key=lambda x: -x[1])[
                :10
            ]:
                print(f"  {dtype}: {count}")
            if stats["arrays"]:
                print(f"\nNumPy arrays:")
                print(f"  Count: {len(stats['arrays'])}")
                print(f"  Total elements: {sum(a['size'] for a in stats['arrays'])}")
            print(f"\nMemory:")
            print(f"  Estimated size: {self._format_bytes(stats['memory'])}")
            print("=" * 60 + "\n")

        return stats

    def _summary_html(self, verbose):
        """Notebook summary with HTML formatting - Dark theme."""
        if not HAS_IPYTHON:
            return self._summary_terminal(verbose)

        stats = self._collect_stats(self)

        html = f"""
        <div style="font-family: 'Courier New', monospace; background: #1e1e1e; 
                    color: #d4d4d4; padding: 20px; border-radius: 5px; 
                    border: 1px solid #3c3c3c;">
            <h3 style="color: #c586c0; border-bottom: 2px solid #3c3c3c; 
                    padding-bottom: 10px; font-weight: normal;">
                📊 Dictionary Summary: {self._name}
            </h3>
            
            <p><strong style="color: #4fc1ff;">Root type:</strong> <span style="color: #d4d4d4;">{type(self).__name__}</span></p>
            <p><strong style="color: #4fc1ff;">Top-level keys:</strong> <span style="color: #d4d4d4;">{len(self)}</span></p>
            
            <h4 style="color: #4ec9b0; margin-top: 20px; font-weight: normal;">Structure</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Total dictionaries:</span> <span style="color: #d4d4d4;">{stats['n_dicts']}</span></li>
                <li><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Total lists/tuples:</span> <span style="color: #d4d4d4;">{stats['n_lists']}</span></li>
                <li><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Total keys:</span> <span style="color: #d4d4d4;">{stats['n_keys']}</span></li>
                <li><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Maximum depth:</span> <span style="color: #d4d4d4;">{stats['max_depth']}</span></li>
            </ul>
            
            <h4 style="color: #4ec9b0; font-weight: normal;">Data types found</h4>
            <ul style="list-style: none; padding-left: 0;">
        """

        for dtype, count in sorted(stats["types"].items(), key=lambda x: -x[1])[:10]:
            html += f'<li><span style="color: #c586c0;">▪</span> <strong style="color: #dcdcaa;">{dtype}:</strong> <span style="color: #d4d4d4;">{count}</span></li>'

        html += "</ul>"

        if stats["arrays"]:
            total_elements = sum(a["size"] for a in stats["arrays"])
            html += f"""
                <h4 style="color: #4ec9b0; font-weight: normal;">NumPy arrays</h4>
                <ul style="list-style: none; padding-left: 0;">
                    <li><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Count:</span> <span style="color: #d4d4d4;">{len(stats['arrays'])}</span></li>
                    <li><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Total elements:</span> <span style="color: #d4d4d4;">{total_elements:,}</span></li>
            """

            if verbose:
                shapes = [a["shape"] for a in stats["arrays"][:5]]
                shapes_str = ", ".join(str(s) for s in shapes)
                if len(stats["arrays"]) > 5:
                    shapes_str += ", ..."
                html += f'<li><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Shapes:</span> <span style="color: #d4d4d4;">{shapes_str}</span></li>'

            html += "</ul>"

        html += f"""
            <h4 style="color: #4ec9b0; font-weight: normal;">Memory</h4>
            <p><span style="color: #569cd6;">▪</span> <span style="color: #9cdcfe;">Estimated size:</span> <strong style="color: #d4d4d4;">{self._format_bytes(stats['memory'])}</strong></p>
        </div>
        """

        display(HTML(html))
        return stats

    def _collect_stats(self, obj, depth=0):
        """Collect detailed statistics recursively."""
        stats = defaultdict(int)
        stats["max_depth"] = depth
        stats["types"] = defaultdict(int)
        stats["arrays"] = []
        stats["memory"] = 0

        def recurse(o, d):
            stats["max_depth"] = max(stats["max_depth"], d)
            type_name = type(o).__name__
            stats["types"][type_name] += 1

            if isinstance(o, dict):
                stats["n_dicts"] += 1
                stats["n_keys"] += len(o)
                for v in o.values():
                    recurse(v, d + 1)
            elif isinstance(o, (list, tuple)):
                stats["n_lists"] += 1
                for item in o:
                    recurse(item, d + 1)
            elif isinstance(o, np.ndarray):
                stats["arrays"].append(
                    {"shape": o.shape, "dtype": o.dtype, "size": o.size}
                )
                stats["memory"] += o.nbytes
            elif isinstance(o, str):
                stats["memory"] += len(o)

        recurse(obj, depth)
        return dict(stats)

    @staticmethod
    def _format_bytes(bytes_size):
        """Format bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} TB"

    def structure(self, max_depth: Optional[int] = 3) -> Dict:
        """
        Get simplified structure showing types and shapes.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to explore

        Returns
        -------
        dict
            Simplified structure representation

        Examples
        --------
        >>> data = ExplorerDict({'a': [1, 2], 'b': {'c': np.array([3, 4])}})
        >>> data.structure(max_depth=2)
        {'a': ['<int>'], 'b': {'c': '<ndarray: shape=(2,), dtype=int64>'}}
        """
        return self._build_structure(self, max_depth, 0)

    def _build_structure(self, obj, max_depth, depth):
        """Recursively build structure representation."""
        if max_depth is not None and depth >= max_depth:
            return self._type_repr(obj)

        if isinstance(obj, dict):
            return {
                key: self._build_structure(val, max_depth, depth + 1)
                for key, val in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return f"<empty {type(obj).__name__}>"
            if len(obj) <= 3:
                return [
                    self._build_structure(item, max_depth, depth + 1) for item in obj
                ]
            else:
                return [
                    self._build_structure(obj[0], max_depth, depth + 1),
                    f"<... {len(obj) - 2} more items ...>",
                    self._build_structure(obj[-1], max_depth, depth + 1),
                ]
        else:
            return self._type_repr(obj)

    def _type_repr(self, obj):
        """Get type representation for leaf nodes."""
        if isinstance(obj, np.ndarray):
            return f"<ndarray: shape={obj.shape}, dtype={obj.dtype}>"
        return f"<{type(obj).__name__}>"

    def search(self, pattern: str, case_sensitive: bool = False) -> List[str]:
        """Search for keys matching a pattern."""
        matches = []

        def recurse(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_str = str(key)
                    current_path = f"{path}.{key}" if path else key

                    if case_sensitive:
                        match = pattern in key_str
                    else:
                        match = pattern.lower() in key_str.lower()

                    if match:
                        matches.append(current_path)

                    recurse(value, current_path)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    recurse(item, f"{path}[{i}]")

        recurse(self)

        if self._notebook_mode and HAS_IPYTHON:
            if matches:
                html = f"""<div style="font-family: monospace; background: #1e1e1e; 
                        color: #d4d4d4; padding: 15px; border-radius: 5px; 
                        border: 1px solid #3c3c3c;">
                    <strong style="color: #4ec9b0;">Found {len(matches)} matches for '{pattern}':</strong>
                    <ul style="padding-left: 20px;">"""
                for match in matches:
                    html += f'<li style="color: #4ec9b0;">{match}</li>'
                html += "</ul></div>"
                display(HTML(html))
            else:
                display(
                    HTML(
                        f'<div style="color: #f48771; font-family: monospace; '
                        f"background: #2d1f1f; padding: 10px; border-radius: 5px; "
                        f"border-left: 4px solid #f48771;\">No matches found for '{pattern}'</div>"
                    )
                )
        else:
            if HAS_CLTCOLORS:
                if matches:
                    print(
                        f"{cltcolors.bcolors.OKGREEN}Found {len(matches)} matches for '{pattern}':{cltcolors.bcolors.ENDC}"
                    )
                    for match in matches:
                        print(
                            f"  {cltcolors.bcolors.DARKCYAN}{match}{cltcolors.bcolors.ENDC}"
                        )
                else:
                    print(
                        f"{cltcolors.bcolors.FAIL}No matches found for '{pattern}'{cltcolors.bcolors.ENDC}"
                    )
            else:
                if matches:
                    print(f"Found {len(matches)} matches for '{pattern}':")
                    for match in matches:
                        print(f"  {match}")
                else:
                    print(f"No matches found for '{pattern}'")

        return matches

    def get_path(self, path: str) -> Any:
        """Get value at a specific path."""
        parts = path.replace("[", ".").replace("]", "").split(".")
        obj = self

        for part in parts:
            if not part:
                continue
            try:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = obj[part]
            except (KeyError, IndexError, TypeError) as e:
                if self._notebook_mode and HAS_IPYTHON:
                    display(
                        HTML(
                            f'<div style="color: #f48771; font-family: monospace; '
                            f"background: #2d1f1f; padding: 10px; border-radius: 5px; "
                            f"border-left: 4px solid #f48771;\">Error accessing '{path}': {e}</div>"
                        )
                    )
                elif HAS_CLTCOLORS:
                    print(
                        f"{cltcolors.bcolors.FAIL}Error accessing '{path}': {e}{cltcolors.bcolors.ENDC}"
                    )
                else:
                    print(f"Error accessing '{path}': {e}")
                return None

        return obj
