"""Share immutable code objects while retaining real source/binding admission."""
import builtins
from contextlib import contextmanager
from functools import lru_cache


@contextmanager
def compilation_fixture():
    original = builtins.compile

    @lru_cache(maxsize=128)
    def cached(source, filename, flags, optimize):
        return original(source,filename,'exec',flags=flags,dont_inherit=True,optimize=optimize)

    def compile_source(source, filename, mode, flags=0, dont_inherit=False, optimize=-1, **kwargs):
        # ASTs are mutable, and inherited future flags belong to the caller.
        # Neither is eligible for this immutable source-compilation fixture.
        if (mode=='exec' and dont_inherit is True and not kwargs and flags==0
                and isinstance(source,(str,bytes)) and isinstance(filename,str)):
            return cached(source,filename,flags,optimize)
        # Noneligible calls retain the caller's future flags when routed
        # through this fixture, rather than inheriting this helper's module.
        if not dont_inherit:
            import __future__,sys
            future = 0
            for name in __future__.all_feature_names:
                future |= getattr(__future__,name).compiler_flag
            flags |= sys._getframe(1).f_code.co_flags & future
            dont_inherit = True
        return original(source,filename,mode,flags=flags,dont_inherit=dont_inherit,optimize=optimize,**kwargs)

    builtins.compile = compile_source
    try:
        yield cached
    finally:
        builtins.compile = original
