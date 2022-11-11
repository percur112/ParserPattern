import base64
import binascii
from collections import OrderedDict
from datetime import date, datetime, tzinfo
from typing import Iterable, List, Type, Union, Set, TypeVar, NewType, Dict

from . import tz
from .lang.STIXPatternParser import STIXPatternParser, ParserRuleContext
from .lang.STIXPatternVisitor import STIXPatternVisitor


ConvertedStixLiteral = NewType('ConvertedStixLiteral', Union[
    int,
    str,
    bool,
    float,
    bytes,
    datetime,
])


class PatternTree(dict):
    

    def __str__(self):
        return self.serialize()

    def serialize(self):
        from .debug import dump_tree
        return dump_tree(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'PatternTree':
       
        if d.keys() != {'pattern'}:
            raise ValueError('Expected a dict with one top-level key, "pattern"')

        tree = PatternTree(d)

        observations = [d['pattern']]
        qualifiers = []
        comparisons = []

        OBSERVATION_TYPES = {
            'expression': cls.format_composite_observation,
            'observation': cls.format_simple_observation,
        }
        while observations:
            node = observations.pop()
            assert len(node) == 1, \
                'Each observation must be a dict with a single key-value pair'

            node_type, body = next(iter(node.items()))
            if node_type not in OBSERVATION_TYPES:
                raise ValueError(
                    f'Unexpected observation type {repr(node_type)}. '
                    f'Expected one of: {OBSERVATION_TYPES.keys()}')

            formatter = OBSERVATION_TYPES[node_type]
            node.update(formatter(**body))
            new_body = next(iter(node.values()))

            if node_type == 'expression':
                observations.extend(new_body['expressions'])
            elif node_type == 'observation':
                comparisons.extend(new_body['expressions'])

            qualifiers.extend(new_body['qualifiers'] or ())

        QUALIFIER_TYPES = {
            'start_stop': cls.format_start_stop_qualifier,
            'within': cls.format_within_qualifier,
            'repeats': cls.format_repeats_qualifier,
        }

        for qualifier in qualifiers:
            assert len(qualifier) == 1, \
                'Each qualifier must be a dict with a single key-value pair'

            qualifier_type, body = next(iter(qualifier.items()))
            if qualifier_type not in QUALIFIER_TYPES:
                raise ValueError(
                    f'Unexpected qualifier type {repr(qualifier_type)}. '
                    f'Expected one of: {QUALIFIER_TYPES.keys()}')

            formatter = QUALIFIER_TYPES[qualifier_type]
            qualifier.update(formatter(**body))

        COMPARISON_TYPES = {
            'expression': cls.format_composite_comparison,
            'comparison': cls.format_simple_comparison,
        }

        while comparisons:
            node = comparisons.pop()
            assert len(node) == 1, \
                'Each comparison must be a dict with a single key-value pair'

            node_type, body = next(iter(node.items()))
            if node_type not in COMPARISON_TYPES:
                raise ValueError(
                    f'Unexpected comparison type {repr(node_type)}. '
                    f'Expected one of: {COMPARISON_TYPES.keys()}')

            formatter = COMPARISON_TYPES[node_type]
            node.update(formatter(**body))
            new_body = next(iter(node.values()))

            if 'expression' in node:
                comparisons.extend(new_body['expressions'])

        return tree

    @classmethod
    def format_pattern(cls, *, root: dict):
        return cls(pattern=root)

    @classmethod
    def format_composite_observation(cls, *,
                                     expressions: List[Dict[str, dict]],
                                     join: str = None,
                                     qualifiers: List[Dict[str, dict]] = None,
                                     ) -> Dict[str, OrderedDict]:
        return {
            'expression': OrderedDict([
                ('join', join),
                ('qualifiers', qualifiers),
                ('expressions', expressions),
            ]),
        }

    @classmethod
    def format_simple_observation(cls, *,
                                  objects: Iterable[str],
                                  expressions: List[Dict[str, dict]],
                                  join: str = None,
                                  qualifiers: List[Dict[str, dict]] = None,
                                  ) -> Dict[str, OrderedDict]:
        if not isinstance(objects, ObjectTypeSet):
            objects = ObjectTypeSet(objects)

        return {
            'observation': OrderedDict([
                ('objects', objects),
                ('join', join),  
                ('qualifiers', qualifiers),
                ('expressions', expressions),
            ]),
        }

    @classmethod
    def format_composite_comparison(cls, *,
                                    expressions: List[Dict[str, dict]],
                                    join: str = None,
                                    ) -> Dict[str, OrderedDict]:
        return {
            'expression': OrderedDict([
                ('join', join),
                ('expressions', expressions),
            ]),
        }

    @classmethod
    def format_simple_comparison(cls, *,
                                 object: str,
                                 path: List[Union[str, slice]],
                                 operator: str,
                                 value: ConvertedStixLiteral,
                                 negated: bool = None,
                                 ) -> Dict[str, OrderedDict]:
        if not isinstance(path, ObjectPath):
           
            path = ObjectPath(path)

        return {
            'comparison': OrderedDict([
                ('object', object),
                ('path', path),
                ('negated', negated),
                ('operator', operator),
                ('value', cls.format_literal(value)),
            ]),
        }

    @classmethod
    def format_start_stop_qualifier(cls, *,
                                    start: datetime,
                                    stop: datetime,
                                    ) -> Dict[str, OrderedDict]:
        return {
            'start_stop': OrderedDict([
                ('start', cls.format_literal(start)),
                ('stop', cls.format_literal(stop)),
            ])
        }

    @classmethod
    def format_within_qualifier(cls, *,
                                value: int,
                                unit: str = 'SECONDS',
                                ) -> Dict[str, OrderedDict]:
        return {
            'within': OrderedDict([
                ('value', cls.format_literal(value)),
                ('unit', unit),
            ])
        }

    @classmethod
    def format_repeats_qualifier(cls, *, value: int) -> Dict[str, OrderedDict]:
        return {
            'repeats': OrderedDict([
                ('value', cls.format_literal(value)),
            ])
        }

    @classmethod
    def format_object_path(cls, *, object: str, path: List[Union[str, slice]]):
        return OrderedDict([
            ('object', object),
            ('path', path),
        ])

    @classmethod
    def format_literal(cls, literal):
        if isinstance(literal, datetime):
            if literal.tzinfo is None:
                literal = literal.replace(tzinfo=tz.utc())
            elif not is_utc(literal):
                raise ValueError('All datetimes must be in UTC')
        return literal


class CompactibleObject:
   

    def is_eligible_for_compaction(self) -> bool:
       
        return False

    def get_literal_type(self) -> Type:
       
        for klass in self.__class__.__mro__:
            if issubclass(klass, CompactibleObject):
                continue

            
            if klass.__module__ == 'typing':
                continue

            return klass

        raise NotImplementedError(
            'This CompactibleObject has no obvious literal type. '
            'Please override the get_literal_type() method.')


class CompactibleList(CompactibleObject, list):
    pass


class CompactibleSet(CompactibleObject, set):
    pass


class ObjectTypeSet(CompactibleSet, Set[str]):
    def is_eligible_for_compaction(self) -> bool:
        return len(self) == 1


class ObjectPath(CompactibleList):
    def is_eligible_for_compaction(self) -> bool:
        return len(self) == 1


class PatternTreeVisitor(STIXPatternVisitor):
    

    def visitPattern(self,
                     ctx: STIXPatternParser.PatternContext,
                     ) -> PatternTree:
        """Convert the root node, Pattern, into a PatternTree"""
        return self.emitPattern(ctx)


    def visitObservationExpressions(self, ctx: STIXPatternParser.ObservationExpressionsContext):
        """Convert <obs_expr> FOLLOWEDBY <obs_expr2> into PatternTree form"""
        return self.emitCompositeObservation(ctx)

    def visitObservationExpressionOr(self, ctx: STIXPatternParser.ObservationExpressionOrContext):
        """Convert <obs_expr> OR <obs_expr2> into PatternTree form"""
        return self.emitCompositeObservation(ctx)

    def visitObservationExpressionAnd(self, ctx: STIXPatternParser.ObservationExpressionAndContext):
        """Convert <obs_expr> AND <obs_expr2> into PatternTree form"""
        return self.emitCompositeObservation(ctx)

    def visitObservationExpressionCompound(self, ctx: STIXPatternParser.ObservationExpressionCompoundContext):
        """Ditch parens around an observation expression"""
        lparen, expr, rparen = ctx.getChildren()
        return self.visit(expr)

   
    def visitObservationExpressionSimple(self, ctx: STIXPatternParser.ObservationExpressionSimpleContext):
        return self.emitSimpleObservation(ctx)

   
    def visitObservationExpressionStartStop(self, ctx: STIXPatternParser.ObservationExpressionStartStopContext):
        return self.emitObservationQualifier(ctx)

    def visitObservationExpressionWithin(self, ctx: STIXPatternParser.ObservationExpressionWithinContext):
        return self.emitObservationQualifier(ctx)

    def visitObservationExpressionRepeated(self, ctx: STIXPatternParser.ObservationExpressionRepeatedContext):
        return self.emitObservationQualifier(ctx)

    def visitStartStopQualifier(self, ctx: STIXPatternParser.StartStopQualifierContext):
        
        start, start_dt, stop, stop_dt = ctx.getChildren()
        return PatternTree.format_start_stop_qualifier(
            start=self.visit(start_dt),
            stop=self.visit(stop_dt),
        )

    def visitWithinQualifier(self, ctx: STIXPatternParser.WithinQualifierContext):
        
        within, number, unit = ctx.getChildren()
        return PatternTree.format_within_qualifier(
            value=self.visit(number),
            unit=self.visit(unit),
        )

    def visitRepeatedQualifier(self, ctx:STIXPatternParser.RepeatedQualifierContext):
        repeats, number, times = ctx.getChildren()
        return PatternTree.format_repeats_qualifier(
            value=self.visit(number),
        )

    def visitComparisonExpression(self, ctx: STIXPatternParser.ComparisonExpressionContext):
        return self.emitCompositeComparison(ctx)

    def visitComparisonExpressionAnd(self, ctx: STIXPatternParser.ComparisonExpressionAndContext):
        return self.emitCompositeComparison(ctx)

  
    def visitPropTestEqual(self, ctx: STIXPatternParser.PropTestEqualContext):
        return self.emitSimpleComparison(ctx)

    def visitPropTestOrder(self, ctx: STIXPatternParser.PropTestOrderContext):
        return self.emitSimpleComparison(ctx)

    def visitPropTestSet(self, ctx: STIXPatternParser.PropTestSetContext):
        return self.emitSimpleComparison(ctx)

    def visitPropTestLike(self, ctx: STIXPatternParser.PropTestLikeContext):
        return self.emitSimpleComparison(ctx)

    def visitPropTestRegex(self, ctx: STIXPatternParser.PropTestRegexContext):
        return self.emitSimpleComparison(ctx)

    def visitPropTestIsSubset(self, ctx: STIXPatternParser.PropTestIsSubsetContext):
        return self.emitSimpleComparison(ctx)

    def visitPropTestIsSuperset(self, ctx: STIXPatternParser.PropTestIsSupersetContext):
        return self.emitSimpleComparison(ctx)

    def visitPropTestParen(self, ctx: STIXPatternParser.PropTestParenContext):
        
        lparen, expr, rparen = ctx.getChildren()
        return self.visit(expr)

    def visitObjectPath(self, ctx: STIXPatternParser.ObjectPathContext):
       
       
        full_path = [self.visit(property)]

        if path:
            path_component: STIXPatternParser.ObjectPathComponentContext = path[0]
            if isinstance(path_component, STIXPatternParser.PathStepContext):
                full_path += self.visit(path_component)
            else:
                full_path.append(self.visit(path_component))

        return PatternTree.format_object_path(
            object=object_type.getText(),
            path=ObjectPath(full_path),
        )

    def visitPathStep(self,
                      ctx: STIXPatternParser.PathStepContext,
                      ) -> List[Union[ConvertedStixLiteral, slice]]:
        
        children = flatten_left(ctx)
        return [
            self.visit(child)
            for child in children
        ]

    def visitIndexPathStep(self, ctx: STIXPatternParser.IndexPathStepContext) -> slice:
      
        lbracket, index, rbracket = ctx.getChildren()
        return slice(self.emitLiteral(index))

    def visitFirstPathComponent(self, ctx: STIXPatternParser.FirstPathComponentContext) -> ConvertedStixLiteral:
       
        return self.emitLiteral(ctx.getChild(0))

    def visitKeyPathStep(self, ctx: STIXPatternParser.KeyPathStepContext) -> ConvertedStixLiteral:
        
        dot, key = ctx.getChildren()
        return self.emitLiteral(key)

    def visitTerminal(self, node) -> ConvertedStixLiteral:
      
        return self.emitLiteral(node)

    def emitPattern(self, ctx: STIXPatternParser.PatternContext):
       
        observations, eof = ctx.getChildren()
        return PatternTree.format_pattern(
            root=self.visit(observations),
        )

    def emitCompositeObservation(self,
                                 ctx: Union[STIXPatternParser.ObservationExpressionsContext,
                                            STIXPatternParser.ObservationExpressionOrContext,
                                            STIXPatternParser.ObservationExpressionAndContext],
       
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))

        op = ctx.getChild(1)
        children = flatten_left(ctx)

        return PatternTree.format_composite_observation(
            join=op.getText().upper(),
            qualifiers=None,  
            expressions=[
               self.visit(child)
               for child in children
           ],
        )

    def emitSimpleObservation(self, ctx: STIXPatternParser.ObservationExpressionSimpleContext):
        
        lbracket, child, rbracket = ctx.getChildren()
        root = self.visit(child)

        if 'expression' in root:
            expression = root['expression']
            join = expression['join']
            expressions = expression['expressions']
        else:
            join = None
            expressions = [root]

        object_types = self.findObjectTypes(expressions)

        return PatternTree.format_simple_observation(
            objects=object_types,
            join=join,
            expressions=expressions,
        )

    def emitObservationQualifier(self, ctx: Union[STIXPatternParser.ObservationExpressionStartStopContext,
                                                  STIXPatternParser.ObservationExpressionWithinContext,
                                                  STIXPatternParser.ObservationExpressionRepeatedContext]):
       
        expr, *qualifiers = flatten_left(ctx, [
            STIXPatternParser.ObservationExpressionStartStopContext,
            STIXPatternParser.ObservationExpressionWithinContext,
            STIXPatternParser.ObservationExpressionRepeatedContext,
        ])

       
        node: dict = self.visit(expr)
        assert isinstance(node, dict) and len(node) == 1  # sanity check

        
        body = next(iter(node.values()))

        body['qualifiers'] = [
            self.visit(qualifier)
            for qualifier in qualifiers
        ]
        return node


    def emitCompositeComparison(self, ctx: Union[STIXPatternParser.ComparisonExpressionContext,
                                                 STIXPatternParser.ComparisonExpressionAndContext]):
      
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))

        op = ctx.getChild(1)
        children = flatten_left(ctx)

        return PatternTree.format_composite_comparison(
            join=op.getText().upper(),
            expressions=[
               self.visit(child)
               for child in children
           ],
        )

    def emitSimpleComparison(self, ctx: STIXPatternParser.PropTestContext) -> dict:
       
        lhs, *nots, op, rhs = ctx.getChildren()

        return PatternTree.format_simple_comparison(
            **self.visit(lhs),
            negated=True if nots else None,  
            operator=op.getText(),
            value=self.visit(rhs),
        )

    def emitLiteral(self, literal) -> ConvertedStixLiteral:
       
        text = literal.getText()
        symbol_type = literal.getSymbol().type
        return coerce_literal(text, symbol_type)

    def findObjectTypes(self, comparison_expressions: List[dict]) -> ObjectTypeSet:
        
        encountered_types = ObjectTypeSet()
        to_visit = list(comparison_expressions)

        while to_visit:
            node = to_visit.pop()
            assert isinstance(node, dict) and len(node) == 1

            node_type, body = next(iter(node.items()))

            if node_type == 'expression':
                to_visit.extend(body['expressions'])
            elif node_type == 'comparison':
                encountered_types.add(body['object'])

        return encountered_types


T = TypeVar('T', bound=ParserRuleContext)


def flatten_left(ctx: ParserRuleContext,
                 rules: Iterable[Type[T]]=None,
                 ) -> List[T]:
    r
    rules = tuple(rules or (type(ctx),))

    flattened = []
    last_lhs = ctx
    while True:
        lhs, *others = last_lhs.getChildren()
        if others:
            flattened.append(others[-1])

        if isinstance(lhs, rules):
            last_lhs = lhs
            continue
        else:
            flattened.append(lhs)
            break

  
    return list(reversed(flattened))



def convert_stix_datetime(timestamp_str: str, ignore_case: bool=False) -> datetime:
    

    if not ignore_case and any(c.islower() for c in timestamp_str):
        raise ValueError(f'Invalid timestamp format (require upper case): {timestamp_str}')

   
    if '.' in timestamp_str:
        fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
    else:
        fmt = '%Y-%m-%dT%H:%M:%SZ'

    dt = datetime.strptime(timestamp_str, fmt)
    dt = dt.replace(tzinfo=tz.utc())

    return dt


def is_utc(o: Union[tzinfo, datetime, date]) -> bool:
   
    if isinstance(o, (datetime, date)):
        if o.tzinfo is None:
            return False
        o = o.tzinfo

    arbitrary_dt = datetime.now().replace(tzinfo=o)
    return arbitrary_dt.utcoffset().total_seconds() == 0



PRIMITIVE_COERCERS = {
    STIXPatternParser.IntPosLiteral: int,
    STIXPatternParser.IntNegLiteral: int,
    STIXPatternParser.StringLiteral: lambda s: s[1:-1].replace("\\'", "'").replace('\\\\', '\\'),
    STIXPatternParser.BoolLiteral: lambda s: s.lower() == 'true',
    STIXPatternParser.FloatPosLiteral: float,
    STIXPatternParser.FloatNegLiteral: float,
    STIXPatternParser.BinaryLiteral: lambda s: base64.standard_b64decode(s[2:-1]),
    STIXPatternParser.HexLiteral: lambda s: binascii.a2b_hex(s[2:-1]),
    STIXPatternParser.TimestampLiteral: lambda t: convert_stix_datetime(t[2:-1]),
}


def coerce_literal(text: str,
                   symbol_type: int,
                   ) -> ConvertedStixLiteral:
    
