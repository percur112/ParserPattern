from typing import Any

import antlr4
import antlr4.error.Errors
import antlr4.error.ErrorListener
import six

from .debug import print_tree
from .lang.STIXPatternLexer import STIXPatternLexer
from .lang.STIXPatternListener import STIXPatternListener
from .lang.STIXPatternParser import STIXPatternParser
from .lang.STIXPatternVisitor import STIXPatternVisitor
from .transform import PatternTreeVisitor, PatternTree


class ParserErrorListener(antlr4.error.ErrorListener.ErrorListener):
    
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.error_message = f'{line}:{column}: {msg}'


class ParseException(Exception):
    """Represents a parse error."""
    pass


def parse(pattern_str: str, trace: bool = False) -> STIXPatternParser.PatternContext:
    """
    Parses the given pattern and returns the antlr parse tree.
    :param pattern_str: The STIX pattern
    :param trace: Whether to enable debug tracing while parsing.
    :return: The parse tree
    :raises ParseException: If there is a parse error
    """
    in_ = antlr4.InputStream(pattern_str)
    lexer = STIXPatternLexer(in_)
    lexer.removeErrorListeners()  
    token_stream = antlr4.CommonTokenStream(lexer)

    parser = STIXPatternParser(token_stream)
    parser.removeErrorListeners() 
    error_listener = ParserErrorListener()
    parser.addErrorListener(error_listener)

    parser._errHandler = antlr4.BailErrorStrategy()

    for i, lit_name in enumerate(parser.literalNames):
        if lit_name == u"<INVALID>":
            parser.literalNames[i] = parser.symbolicNames[i]

    parser.setTrace(trace)

    try:
        tree = parser.pattern()
    except antlr4.error.Errors.ParseCancellationException as e:
        
        real_exc = e.args[0]

        parser._errHandler.reportError(parser, real_exc)

        six.raise_from(ParseException(error_listener.error_message),
                       real_exc)
    else:
        return tree


class Pattern:
    """A parsed pattern expression, with traversal and representation methods
    """
    tree: STIXPatternParser.PatternContext

    def __init__(self, pattern_str: str):
        """
        Compile a pattern.

        """
        self.tree = parse(pattern_str)

    def walk(self, listener: STIXPatternListener):
        
        antlr4.ParseTreeWalker.DEFAULT.walk(listener, self.tree)

    def visit(self, visitor: STIXPatternVisitor) -> Any:
       
        return self.tree.accept(visitor)

    def to_dict_tree(self) -> PatternTree:
       
        visitor = PatternTreeVisitor()
        return visitor.visit(self.tree)

    def print_dict_tree(self):
        
        print_tree(self.to_dict_tree())
