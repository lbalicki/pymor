# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module provides base classes from which most classes in pyMOR inherit.

The purpose of these classes is to provide some common functionality for
all objects in pyMOR. The most notable features provided by :class:`BasicInterface`
are the following:

    1. :class:`BasicInterface` sets class :class:`UberMeta` as metaclass
       which itself inherits from :class:`abc.ABCMeta`. Thus it is possible
       to define interface classes with abstract methods using the
       :func:`abstractmethod` decorator. There are also decorators for
       abstract class methods, static methods, and properties.
    2. Using metaclass magic, each *class* deriving from :class:`BasicInterface`
       comes with its own :mod:`~pymor.core.logger` instance accessible through its `logger`
       attribute. The logger prefix is automatically set to the class name.
    3. Logging can be disabled and re-enabled for each *instance* using the
       :meth:`BasicInterface.disable_logging` and :meth:`BasicInterface.enable_logging`
       methods.
    4. :meth:`BasicInterface.uid` provides a unique id for each instance. While
       `id(obj)` is only guaranteed to be unique among all living Python objects,
       :meth:`BasicInterface.uid` will be (almost) unique among all pyMOR objects
       that have ever existed, including previous runs of the application. This
       is achieved by building the id from a uuid4 which is newly created for
       each pyMOR run and a counter which is increased for any object that requests
       an uid.
    5. If not set by the user to another value, :attr:`BasicInterface.name` is
       set to the name of the object's class.


:class:`ImmutableInterface` derives from :class:`BasicInterface` and adds the following
functionality:

    1. Using more metaclass magic, each instance which derives from
       :class:`ImmutableInterface` is locked after its `__init__` method has returned.
       Each attempt to change one of its attributes raises an exception. Private
       attributes (of the form `_name`) are exempted from this rule.
    2. A unique _`state id` for the instance can be calculated by calling
       :meth:`~ImmutableInterface.generate_sid` and is then stored as the object's
       `sid` attribute.
       The state id is obtained by deterministically serializing the object's state
       and then computing a checksum of the resulting byte stream.
    3. :attr:`ImmutableInterface.sid_ignore` can be set to a set of attribute names
       which should be excluded from state id calculation.
    4. :meth:`ImmutableInterface.with_` can be used to create a copy of an instance with
       some changed attributes. E.g. ::

           obj.with_(a=x, b=y)

       creates a copy with the `a` and `b` attributes of `obj` set to `x` and `y`.
       `with_` is implemented by creating a new instance, passing the arguments of
       `with_` to `__init__`. The missing `__init__` arguments are taken from instance
       attributes of the same name.
"""

import abc
try:
    from cPickle import dumps, HIGHEST_PROTOCOL
except ImportError:
    from pickle import dumps, HIGHEST_PROTOCOL
from copyreg import dispatch_table
from functools import wraps
import hashlib
import inspect
import itertools
import os
import time
from types import FunctionType, BuiltinFunctionType
import uuid

import numpy as np

from pymor.core import logger
from pymor.core.exceptions import ConstError, SIDGenerationError
from pymor.tools.formatrepr import format_repr, _format_generic

DONT_COPY_DOCSTRINGS = int(os.environ.get('PYMOR_WITH_SPHINX', 0)) == 1
NoneType = type(None)


class UID:
    '''Provides unique, quickly computed ids by combining a session UUID4 with a counter.'''

    __slots__ = ['uid']

    prefix = f'{uuid.uuid4()}_'
    counter = [0]

    def __init__(self):
        self.uid = self.prefix + str(self.counter[0])
        self.counter[0] += 1

    def __getstate__(self):
        return 1

    def __setstate__(self, v):
        self.uid = self.prefix + str(self.counter[0])
        self.counter[0] += 1


class UberMeta(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        """Metaclass of :class:`BasicInterface`.

        I tell base classes when I derive a new class from them. I create a logger
        for each class I create. I add an `init_args` attribute to the class.
        """

        # all bases except object get the derived class' name appended
        for base in [b for b in bases if b != object]:
            derived = cls
            # mangle the name to the base scope
            attribute = '_%s__implementors' % base.__name__
            if hasattr(base, attribute):
                getattr(base, attribute).append(derived)
            else:
                setattr(base, attribute, [derived])
        cls._logger = logger.getLogger(f'{cls.__module__.replace("__main__", "pymor")}.{name}')
        abc.ABCMeta.__init__(cls, name, bases, namespace)

    def __new__(cls, classname, bases, classdict):
        """I copy docstrings from base class methods to deriving classes.

        Copying of docstrings is disabled when the `PYMOR_WITH_SPHINX` environment
        variable is set to `1`.
        """
        for attr in ('_init_arguments', '_init_defaults'):
            if attr in classdict:
                raise ValueError(attr + ' is a reserved class attribute for subclasses of BasicInterface')

        for attr, item in classdict.items():
            if isinstance(item, FunctionType):
                # first copy docs
                base_doc = None
                for base in bases:
                    base_func = getattr(base, item.__name__, None)
                    if not DONT_COPY_DOCSTRINGS:
                        if base_func:
                            base_doc = getattr(base_func, '__doc__', None)
                        if base_doc:
                            doc = getattr(item, '__doc__', '')
                            if doc is not None:
                                base_doc = doc
                            item.__doc__ = base_doc

        def __auto_init(self, locals_):
            """Automatically assign __init__ arguments.

            This method is used in __init__ to automatically assign __init__ arguments to equally
            named object attributes. The values are provided by the `locals_` dict. Usually,
            `__auto_init` is called as::

                self.__auto_init(locals())

            where `locals()` returns a dictionary of all local variables in the current scope.
            Only attributes which have not already been set by the user are initialized by
            `__auto_init`.
            """
            for arg in c._init_arguments:
                if arg not in self.__dict__:
                    setattr(self, arg, locals_[arg])

        classdict[f'_{classname}__auto_init'] = __auto_init

        c = abc.ABCMeta.__new__(cls, classname, bases, classdict)

        # getargspec is deprecated and does not work with keyword only args
        init_sig = inspect.signature(c.__init__)
        init_args = []
        for arg, description in init_sig.parameters.items():
            if arg == 'self':
                continue
            if description.kind == description.POSITIONAL_ONLY:
                raise TypeError('It should not be possible that {}.__init__ has POSITIONAL_ONLY arguments'.
                                format(c))
            if description.kind in (description.POSITIONAL_OR_KEYWORD, description.KEYWORD_ONLY):
                init_args.append(arg)
        c._init_arguments = tuple(init_args)

        return c


class BasicInterface(metaclass=UberMeta):
    """Base class for most classes in pyMOR.

    Attributes
    ----------
    logger
        A per-class instance of :class:`logging.Logger` with the class
        name as prefix.
    logging_disabled
        `True` if logging has been disabled.
    name
        The name of the instance. If not set by the user, the name is
        set to the class name.
    uid
        A unique id for each instance. The uid is obtained by using
        :class:`UID` and is unique for all pyMOR objects ever created.
    """
    @property
    def name(self):
        n = getattr(self, '_name', None)
        return n or type(self).__name__

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def logging_disabled(self):
        return self._logger is logger.dummy_logger

    @property
    def logger(self):
        return self._logger

    def disable_logging(self, doit=True):
        """Disable logging output for this instance."""
        if doit:
            self._logger = logger.dummy_logger
        else:
            del self._logger

    def enable_logging(self, doit=True):
        """Enable logging output for this instance."""
        self.disable_logging(not doit)

    @classmethod
    def implementors(cls, descend=False):
        """I return a, potentially empty, list of my subclass-objects.
        If `descend` is `True`, I traverse my entire subclass hierarchy and return a flattened list.
        """
        if not hasattr(cls, '_%s__implementors' % cls.__name__):
            return []
        level = getattr(cls, '_%s__implementors' % cls.__name__)
        if not descend:
            return level
        subtrees = itertools.chain.from_iterable([sub.implementors() for sub in level if sub.implementors() != []])
        level.extend(subtrees)
        return level

    @classmethod
    def implementor_names(cls, descend=False):
        """For convenience I return a list of my implementor names instead of class objects"""
        return [c.__name__ for c in cls.implementors(descend)]

    @classmethod
    def has_interface_name(cls):
        """`True` if the class name ends with `Interface`. Used for introspection."""
        name = cls.__name__
        return name.endswith('Interface')

    _uid = None

    @property
    def uid(self):
        if self._uid is None:
            self._uid = UID()
        return self._uid.uid

    def _format_repr(self, max_width, verbosity, override={}):
        if verbosity < 3 and self.name == type(self).__name__ and 'name' not in override:
            override = dict(override, name=None)
        return _format_generic(self, max_width, verbosity, override=override)

    def __repr__(self):
        return format_repr(self)


abstractmethod = abc.abstractmethod
abstractproperty = abc.abstractproperty
abstractclassmethod = abc.abstractclassmethod
abstractstaticmethod = abc.abstractstaticmethod


class classinstancemethod:

    def __init__(self, cls_meth):
        self.cls_meth = cls_meth

    def __get__(self, instance, cls):
        if cls is None:
            return self
        if instance is None:
            @wraps(self.cls_meth)
            def the_class_method(*args, **kwargs):
                return self.cls_meth(cls, *args, **kwargs)
            return the_class_method
        else:
            @wraps(self.inst_meth)
            def the_instance_method(*args, **kwargs):
                return self.inst_meth(instance, *args, **kwargs)
            return the_instance_method

    def instancemethod(self, inst_meth):
        inst_meth.__doc__ = inst_meth.__doc__ or self.cls_meth.__doc__
        self.inst_meth = inst_meth
        return self


class ImmutableMeta(UberMeta):
    """Metaclass for :class:`ImmutableInterface`."""

    def __new__(cls, classname, bases, classdict):

        # Ensure that '_sid_contains_cycles' and 'sid' are contained in sid_ignore.
        # Otherwise sids of objects in reference cycles may depend on the order in which
        # generate_sid is called upon these objects.
        if 'sid_ignore' in classdict:
            classdict['sid_ignore'] = set(classdict['sid_ignore']) | {'_sid_contains_cycles', 'sid'}

        c = UberMeta.__new__(cls, classname, bases, classdict)

        c._implements_reduce = ('__reduce__' in classdict
                                or '__reduce_ex__' in classdict
                                or any(getattr(base, '_implements_reduce', False)
                                       for base in bases))

        # set __signature__ attribute on newly created class c to ensure that
        # inspect.signature(c) returns the signature of its __init__ arguments and not
        # the signature of ImmutableMeta.__call__
        sig = inspect.signature(c.__init__)
        c.__signature__ = sig.replace(parameters=tuple(sig.parameters.values())[1:])
        return c

    def _call(self, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        assert all(hasattr(instance, arg) for arg in instance._init_arguments), \
            (f'__init__ arguments {[arg for arg in instance._init_arguments if not hasattr(instance,arg)]} '
             f'of class {self.__name__} not available as instance attributes\n'
             f'(all __init__ args need to be attributes for with_ to work).')
        instance._locked = True
        return instance

    __call__ = _call


class ImmutableInterface(BasicInterface, metaclass=ImmutableMeta):
    """Base class for immutable objects in pyMOR.

    Instances of `ImmutableInterface` are immutable in the sense that
    after execution of `__init__`, any modification of a non-private
    attribute will raise an exception.

    .. _ImmutableInterfaceWarning:
    .. warning::
           For instances of `ImmutableInterface`,
           the result of member function calls should be completely
           determined by the function's arguments together with the
           object's |state id| and the current state of pyMOR's
           global |defaults|.

    While, in principle, you are allowed to modify private members after
    instance initialization, this should never affect the outcome of
    future method calls. In particular, if you update any internal state
    after initialization, you have to ensure that this state is not affected
    by possible changes of the global :mod:`~pymor.core.defaults`.

    Also note that mutable private attributes will cause false cache
    misses when these attributes enter |state id| calculation. If your
    implementation uses such attributes, you should therefore add their
    names to the :attr:`~ImmutableInterface.sid_ignore` set.

    Attributes
    ----------
    sid
        The objects |state id|. Only available after
        :meth:`~ImmutableInterface.generate_sid` has been called.
    sid_ignore
        Set of attributes not to include in |state id| calculation.
    """
    sid_ignore = frozenset({'_locked', '_logger', '_name', '_uid', '_sid_contains_cycles', 'sid'})

    _locked = False

    # we need to define __init__, otherwise the Python 2 signature hack will fail
    def __init__(self):
        pass

    def __setattr__(self, key, value):
        """depending on _locked state I delegate the setattr call to object or
        raise an Exception
        """
        if not self._locked or key[0] == '_':
            return object.__setattr__(self, key, value)
        else:
            raise ConstError(f'Changing "{key}" is not allowed in locked "{self.__class__}"')

    def generate_sid(self, debug=False):
        """Generate a unique |state id| for the given object.

        The generated state id is stored in the object's `sid` attribute.

        Parameters
        ----------
        debug
            If `True`, produce some debugging output.

        Returns
        -------
        The generated |state id|.
        """
        if hasattr(self, 'sid'):
            return self.sid
        else:
            return self._generate_sid(debug, ())

    def _generate_sid(self, debug, seen_immutables):
        sid_generator = _SIDGenerator()
        sid, has_cycles = sid_generator.generate(self, debug, seen_immutables)
        self.__dict__['sid'] = sid
        self.__dict__['_sid_contains_cycles'] = has_cycles
        return sid

    def with_(self, new_type=None, **kwargs):
        """Returns a copy with changed attributes.

        A a new class instance is created with the given keyword arguments as
        arguments for `__init__`.  Missing arguments are obtained form instance
        attributes with the
        same name.

        Parameters
        ----------
        new_type
            If not None, return an instance of this class (instead of `type(self)`).
        `**kwargs`
            Names of attributes to change with their new values. Each attribute name
            has to be an argument to `__init__`.

        Returns
        -------
        Copy of `self` with changed attributes.
        """
        # fill missing __init__ arguments using instance attributes of same name
        for arg in (self._init_arguments if new_type is None else new_type._init_arguments):
            if arg not in kwargs:
                try:
                    kwargs[arg] = getattr(self, arg)
                except AttributeError:
                    raise ValueError(f"Cannot find missing __init__ argument '{arg}' for '{self.__class__}' "
                                     f"as attribute of '{self}'")

        c = (type(self) if new_type is None else new_type)(**kwargs)

        if self.logging_disabled:
            c.disable_logging()

        return c

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def generate_sid(obj, debug=False):
    """Generate a unique |state id| for the current state of the given object.

    Parameters
    ----------
    obj
        The object for which to compute the state sid.
    debug
        If `True`, produce some debug output.

    Returns
    -------
    The generated state id.
    """
    sid_generator = _SIDGenerator()
    return sid_generator.generate(obj, debug, ())[0]


# Helper classes for generate_sid

STRING_TYPES = (str, bytes)


class _SIDGenerator:

    def __init__(self):
        self.memo = {}
        self.logger = logger.getLogger('pymor.core.interfaces')

    def generate(self, obj, debug, seen_immutables):
        start = time.time()

        self.has_cycles = False
        self.seen_immutables = seen_immutables + (id(obj),)
        self.debug = debug
        state = self.deterministic_state(obj, first_obj=True)

        if debug:
            print('-' * 100)
            print('Deterministic state for ' + getattr(obj, 'name', str(obj)))
            print('-' * 100)
            print()
            import pprint
            pprint.pprint(state, indent=4)
            print()

        sid = hashlib.sha256(dumps(state, protocol=-1)).hexdigest()

        if debug:
            print(f'SID: {sid}, reference cycles: {self.has_cycles}')
            print()
            print()

        name = getattr(obj, 'name', None)
        if name:
            self.logger.debug(f'{name}: SID generation took {time.time()-start} seconds')
        else:
            self.logger.debug(f'SID generation took {time.time()-start} seconds')
        return sid, self.has_cycles

    def deterministic_state(self, obj, first_obj=False):
        v = self.memo.get(id(obj))
        if v:
            return(v)

        t = type(obj)
        if t in (NoneType, bool, int, float, FunctionType, BuiltinFunctionType, type):
            return obj

        self.memo[id(obj)] = _MemoKey(len(self.memo), obj)

        if t in STRING_TYPES:
            return obj

        if t is np.ndarray and t.dtype != object:
            return obj

        if t is tuple:
            return (tuple,) + tuple(self.deterministic_state(x) for x in obj)

        if t is list:
            return [self.deterministic_state(x) for x in obj]

        if t in (set, frozenset):
            return (t,) + tuple(self.deterministic_state(x) for x in sorted(obj))

        if t is dict:
            return (dict,) + tuple((k if type(k) is str else self.deterministic_state(k), self.deterministic_state(v))
                                   for k, v in sorted(obj.items()))

        if issubclass(t, ImmutableInterface):
            if hasattr(obj, 'sid') and not obj._sid_contains_cycles:
                return (t, obj.sid)

            if not first_obj:
                if id(obj) in self.seen_immutables:
                    raise _SIDGenerationRecursionError
                try:
                    obj._generate_sid(self.debug, self.seen_immutables)
                    return (t, obj.sid)
                except _SIDGenerationRecursionError:
                    self.has_cycles = True
                    self.logger.debug(f'{obj.name}: contains cycles of immutable objects, consider refactoring')

            if obj._implements_reduce:
                self.logger.debug(f'{obj.name}: __reduce__ is implemented, not using sid_ignore')
                return self.handle_reduce_value(obj, t, obj.__reduce_ex__(HIGHEST_PROTOCOL), first_obj)
            else:
                try:
                    state = obj.__getstate__()
                except AttributeError:
                    state = obj.__dict__
                state = {k: v for k, v in state.items() if k not in obj.sid_ignore}
                return self.deterministic_state(state) if first_obj else (t, self.deterministic_state(state))

        sid = getattr(obj, 'sid', None)
        if sid:
            return sid if first_obj else (t, sid)

        reduce = dispatch_table.get(t)
        if reduce:
            rv = reduce(obj)
        else:
            if issubclass(t, type):
                return obj

            reduce = getattr(obj, '__reduce_ex__', None)
            if reduce:
                rv = reduce(HIGHEST_PROTOCOL)
            else:
                reduce = getattr(obj, '__reduce__', None)
                if reduce:
                    rv = reduce()
                else:
                    raise SIDGenerationError(f'Cannot handle {obj} of type {t.__name__}')

        return self.handle_reduce_value(obj, t, rv, first_obj)

    def handle_reduce_value(self, obj, t, rv, first_obj):
        if type(rv) is str:
            raise SIDGenerationError('__reduce__ methods returning a string are currently not handled '
                                     + f'(object {obj} of type {t.__name__})')

        if type(rv) is not tuple or not (2 <= len(rv) <= 5):
            raise SIDGenerationError(f'__reduce__ return value malformed (object {obj} of type {t.__name__})')

        rv = rv + (None,) * (5 - len(rv))
        func, args, state, listitems, dictitems = rv

        state = (func,
                 tuple(self.deterministic_state(x) for x in args),
                 self.deterministic_state(state),
                 self.deterministic_state(tuple(listitems)) if listitems is not None else None,
                 self.deterministic_state(sorted(dictitems)) if dictitems is not None else None)

        return state if first_obj else (t,) + state


class _MemoKey:
    def __init__(self, key, obj):
        self.key = key
        self.obj = obj

    def __repr__(self):
        return f'_MemoKey({self.key}, {repr(self.obj)})'

    def __getstate__(self):
        return self.key


class _SIDGenerationRecursionError(Exception):
    pass
