"""
Debug wrapper for dotplot to diagnose rendering issues
"""

import matplotlib
import matplotlib.pyplot as plt

from .results import dotplot as _original_dotplot


def debug_dotplot(*args, **kwargs):
    """
    Debug wrapper for dotplot that provides diagnostic information
    """
    print("=" * 60)
    print("DOTPLOT DEBUG MODE")
    print("=" * 60)

    # Check matplotlib state
    print("\n1. Matplotlib State:")
    print(f"   Backend: {matplotlib.get_backend()}")
    print(f"   Interactive: {matplotlib.is_interactive()}")
    print(f"   Current figures: {plt.get_fignums()}")

    # Call original dotplot
    print("\n2. Creating dotplot...")
    try:
        fig = _original_dotplot(*args, **kwargs)
        print("   ✓ Figure created successfully")
        print(f"   Type: {type(fig)}")
        print(f"   Number: {fig.number if hasattr(fig, 'number') else 'N/A'}")
        print(f"   Size: {fig.get_figwidth():.1f} x {fig.get_figheight():.1f}")
        print(f"   DPI: {fig.dpi}")

        # Check axes content
        if fig.axes:
            ax = fig.axes[0]
            print("\n3. Axes Content:")
            print(f"   Number of axes: {len(fig.axes)}")
            print(f"   Collections: {len(ax.collections)}")
            print(f"   Texts: {len(ax.texts)}")
            print(f"   Lines: {len(ax.lines)}")
            print(f"   Title: '{ax.get_title()}'")
            print(f"   X-label: '{ax.get_xlabel()}'")
            print(f"   Y-label: '{ax.get_ylabel()}'")

            # Check if scatter plot has data
            if ax.collections:
                for i, coll in enumerate(ax.collections):
                    if hasattr(coll, "get_offsets"):
                        offsets = coll.get_offsets()
                        print(f"   Collection {i}: {len(offsets)} points")
        else:
            print("\n3. WARNING: No axes found in figure!")

        # Try different display methods
        print("\n4. Display Attempts:")

        # Method 1: Check if we're in IPython/Jupyter
        try:
            from IPython import get_ipython
            from IPython.display import display

            ipython = get_ipython()
            if ipython:
                print(f"   IPython detected: {ipython.__class__.__name__}")

                # Check matplotlib mode
                try:
                    magic = ipython.magic("matplotlib")
                    print(f"   Matplotlib mode: {magic}")
                except:
                    print("   Could not determine matplotlib mode")

                # Try display
                print("   Attempting IPython display()...")
                display(fig)
                print("   ✓ display() called")
        except:
            print("   Not in IPython/Jupyter environment")

        # Method 2: Standard plt.show()
        print("   Attempting plt.show()...")
        plt.show()
        print("   ✓ plt.show() called")

        # Provide guidance
        print("\n5. If the plot didn't appear, try:")
        print("   a) Run this in a new cell: %matplotlib inline")
        print("   b) Run this in a new cell: plt.show()")
        print("   c) In a new cell, just type: fig")
        print("   d) Save and display as image:")
        print("      fig.savefig('test.png')")
        print("      from IPython.display import Image")
        print("      Image('test.png')")

        print("\n" + "=" * 60)
        return fig

    except Exception as e:
        print(f"   ✗ Error creating dotplot: {e}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 60)
        raise


def check_matplotlib_setup():
    """
    Check matplotlib setup and provide recommendations
    """
    print("Checking Matplotlib Setup")
    print("-" * 40)

    # Basic info
    print(f"Version: {matplotlib.__version__}")
    print(f"Backend: {matplotlib.get_backend()}")
    print(f"Interactive: {matplotlib.is_interactive()}")
    print(f"Config dir: {matplotlib.get_configdir()}")

    # Check for common backends
    backends_to_try = ["TkAgg", "Qt5Agg", "MacOSX", "notebook", "inline"]
    print("\nAvailable backends:")
    for backend in backends_to_try:
        try:
            matplotlib.use(backend, force=False)
            print(f"  ✓ {backend}")
        except:
            print(f"  ✗ {backend}")

    # Reset to original
    matplotlib.use(matplotlib.get_backend(), force=True)

    # Check for Jupyter
    try:
        get_ipython()
        print("\n✓ Running in IPython/Jupyter")
        print("  Recommended: Run '%matplotlib inline' in a cell")
    except:
        print("\n✗ Not in IPython/Jupyter")
        print("  Plots should appear in separate windows")

    print("-" * 40)
