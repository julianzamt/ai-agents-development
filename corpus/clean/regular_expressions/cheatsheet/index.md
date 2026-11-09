---
title: Regular expression syntax cheat sheet
slug: Web/JavaScript/Guide/Regular_expressions/Cheatsheet
---

This page provides an overall cheat sheet of all the capabilities of `RegExp` syntax by aggregating the content of the articles in the `RegExp` guide. If you need more information on a specific topic, please follow the link on the corresponding heading to access the full article or head to the guide.

## Character classes

Character classes distinguish kinds of characters such as, for example, distinguishing between letters and digits.

| Characters | Meaning |
| --- | --- |
| `[xyz] [a-c]` | **Character class:** Matches any one of the enclosed characters. You can specify a range of characters by using a hyphen, but if the hyphen appears as the first or last character enclosed in the square brackets, it is taken as a literal hyphen to be included in the character class as a normal character. For example, `[abcd]` is the same as `[a-d]`. They match the "b" in "brisket", and the "c" in "chop". For example, `[abcd-]` and `[-abcd]` match the "b" in "brisket", the "c" in "chop", and the "-" (hyphen) in "non-profit". For example, `[\w-]` is the same as `[A-Za-z0-9_-]`. They both match the "b" in "brisket", the "c" in "chop", and the "n" in "non-profit". |
| `[^xyz] [^a-c]` | **Negated character class:** Matches anything that is not enclosed in the square brackets. You can specify a range of characters by using a hyphen, but if the hyphen appears as the first character after the `^` or the last character enclosed in the square brackets, it is taken as a literal hyphen to be included in the character class as a normal character. For example, `[^abc]` is the same as `[^a-c]`. They initially match "o" in "bacon" and "h" in "chop". **Note:** The ^ character may also indicate the beginning of input. |
| `.` | **Wildcard:** Matches any single character *except* line terminators: `\n`, `\r`, `\u2028` or `\u2029`. For example, `/.y/` matches "my" and "ay", but not "yes", in "yes make my day", as there is no character before "y" in "yes". If the `dotAll` (s) flag is enabled, also matches line terminators. Inside a character class, the dot loses its special meaning and matches a literal dot. |
| `\d` | **Digit character class escape:** Matches any digit (Arabic numeral). Equivalent to `[0-9]`. For example, `/\d/` or `/[0-9]/` matches "2" in "B2 is the suite number". |
| `\D` | **Non-digit character class escape:** Matches any character that is not a digit (Arabic numeral). Equivalent to `[^0-9]`. For example, `/\D/` or `/[^0-9]/` matches "B" in "B2 is the suite number". |
| `\w` | **Word character class escape:** Matches any alphanumeric character from the basic Latin alphabet, including the underscore. Equivalent to `[A-Za-z0-9_]`. For example, `/\w/` matches "a" in "apple", "5" in "$5.28", "3" in "3D" and "m" in "Émanuel". |
| `\W` | **Non-word character class escape:** Matches any character that is not a word character from the basic Latin alphabet. Equivalent to `[^A-Za-z0-9_]`. For example, `/\W/` or `/[^A-Za-z0-9_]/` matches "%" in "50%" and "É" in "Émanuel". |
| `\s` | **White space character class escape:** Matches a single white space character, including space, tab, form feed, line feed, and other Unicode spaces. Equivalent to `[\f\n\r\t\v\u0020\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]`. For example, `/\s\w*/` matches " bar" in "foo bar". |
| `\S` | **Non-white space character class escape:** Matches a single character other than white space. Equivalent to `[^\f\n\r\t\v\u0020\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]`. For example, `/\S\w*/` matches "foo" in "foo bar". |
| `\t` | Matches a horizontal tab. |
| `\r` | Matches a carriage return. |
| `\n` | Matches a linefeed. |
| `\v` | Matches a vertical tab. |
| `\f` | Matches a form-feed. |
| `[\b]` | Matches a backspace. If you're looking for the word-boundary assertion (`\b`), see Assertions. |
| `\0` | Matches a NUL character. Do not follow this with another digit. |
| `\c*X*` | Matches a control character using caret notation, where "X" is a letter from A–Z or a–z (corresponding to code points `U+0001`*–*`U+001A`). For example, `/\cM\cJ/` matches "\r\n". |
| `\x*hh*` | **Hex escape:** Matches the character with the code `*hh*` (two hexadecimal digits). |
| `\u*HHHH*` | **Unicode escape:** Matches a UTF-16 code-unit with the value `*HHHH*` (four hexadecimal digits). |
| `\u{*H…H*}` | **Unicode code point escape:** Matches the character with the Unicode value `U+*H…H*` (1 to 6 hexadecimal digits). Only valid in Unicode-aware mode. |
| `\p{*UnicodeProperty*}`, `\P{*UnicodeProperty*}` | **Unicode character class escape:** Matches a character based on its Unicode character properties: for example, emoji characters, or Japanese *katakana* characters, or Chinese/Japanese Han/Kanji characters, etc.). Only valid in Unicode-aware mode. |
| `\` | Indicates that the following character should be treated specially, or "escaped". It behaves one of two ways. For characters that are usually treated literally, indicates that the next character is special and not to be interpreted literally. For example, `/b/` matches the character "b". By placing a backslash in front of "b", that is by using `/\b/`, the character becomes special to mean match a word boundary. For characters that are usually treated specially, indicates that the next character is not special and should be interpreted literally. For example, "*" is a special character that means 0 or more occurrences of the preceding character should be matched; for example, `/a*/` means match 0 or more "a"s. To match `*` literally, precede it with a backslash; for example, `/a\*/` matches "a*". **Note:** To match this character literally, escape it with itself. In other words to search for `\` use `/\\/`. |
| `*x*\|*y*` | **Disjunction:** Matches either "x" or "y". Each component, separated by a pipe (`\|`), is called an *alternative*. For example, `/green\|red/` matches "green" in "green apple" and "red" in "red apple". **Note:** A disjunction is another way to specify "a set of choices", but it's not a character class. Disjunctions are not atoms — you need to use a group to make it part of a bigger pattern. `[abc]` is functionally equivalent to `(?:a\|b\|c)`. |

## Assertions

Assertions include boundaries, which indicate the beginnings and endings of lines and words, and other patterns indicating in some way that a match is possible (including look-ahead, look-behind, and conditional expressions).

### Boundary-type assertions

| Characters | Meaning |
| --- | --- |
| `^` | **Input boundary beginning assertion:** Matches the beginning of input. If the `multiline` (m) flag is enabled, also matches immediately after a line break character. For example, `/^A/` does not match the "A" in "an A", but does match the first "A" in "An A". **Note:** This character has a different meaning when it appears at the start of a character class. |
| `$` | **Input boundary end assertion:** Matches the end of input. If the `multiline` (m) flag is enabled, also matches immediately before a line break character. For example, `/t$/` does not match the "t" in "eater", but does match it in "eat". |
| `\A` | **Buffer boundary start assertion:** Matches the start of the entire string regardless of the presence of the `m` flag. Only valid in Unicode-aware mode. |
| `\z` | **Buffer boundary end assertion:** Matches the end of the entire string regardless of the presence of the `m` flag. Only valid in Unicode-aware mode. |
| `\Z` | **Buffer boundary end assertion with optional newline:** Matches the end of the entire string, but allows an optional trailing newline character sequence (either a line terminator or a `\r\n` sequence). Only valid in Unicode-aware mode. |
| `\b` | **Word boundary assertion:** Matches a word boundary. This is the position where a word character is not followed or preceded by another word-character, such as between a letter and a space. Note that a matched word boundary is not included in the match. In other words, the length of a matched word boundary is zero. Examples: `/\bm/` matches the "m" in "moon". `/oo\b/` does not match the "oo" in "moon", because "oo" is followed by "n" which is a word character. `/oon\b/` matches the "oon" in "moon", because "oon" is the end of the string, thus not followed by a word character. `/\w\b\w/` will never match anything, because a word character can never be followed by both a non-word and a word character. To match a backspace character (`[\b]`), see Character Classes. |
| `\B` | **Non-word-boundary assertion:** Matches a non-word boundary. This is a position where the previous and next character are of the same type: Either both must be words, or both must be non-words, for example between two letters or between two spaces. The beginning and end of a string are considered non-words. Same as the matched word boundary, the matched non-word boundary is also not included in the match. For example, `/\Bon/` matches "on" in "at noon", and `/ye\B/` matches "ye" in "possibly yesterday". |

### Other assertions

> **Note:**
> The `?` character may also be used as a quantifier.

| Characters | Meaning |
| --- | --- |
| `x(?=y)` | **Lookahead assertion:** Matches "x" only if "x" is followed by "y". For example, `/Jack(?=Sprat)/` matches "Jack" only if it is followed by "Sprat". `/Jack(?=Sprat\|Frost)/` matches "Jack" only if it is followed by "Sprat" or "Frost". However, neither "Sprat" nor "Frost" is part of the match results. |
| `x(?!y)` | **Negative lookahead assertion:** Matches "x" only if "x" is not followed by "y". For example, `/\d+(?!\.)/` matches a number only if it is not followed by a decimal point. `/\d+(?!\.)/.exec('3.141')` matches "141" but not "3". |
| `(?<=y)x` | **Lookbehind assertion:** Matches "x" only if "x" is preceded by "y". For example, `/(?<=Jack)Sprat/` matches "Sprat" only if it is preceded by "Jack". `/(?<=Jack\|Tom)Sprat/` matches "Sprat" only if it is preceded by "Jack" or "Tom". However, neither "Jack" nor "Tom" is part of the match results. |
| `(?<!y)x` | **Negative lookbehind assertion:** Matches "x" only if "x" is not preceded by "y". For example, `/(?<!-)\d+/` matches a number only if it is not preceded by a minus sign. `/(?<!-)\d+/.exec('3')` matches "3". `/(?<!-)\d+/.exec('-3')` match is not found because the number is preceded by the minus sign. |

## Groups and backreferences

Groups and backreferences indicate groups of expression characters.

| Characters | Meaning |
| --- | --- |
| `(*x*)` | **Capturing group:** Matches `*x*` and remembers the match. For example, `/(foo)/` matches and remembers "foo" in "foo bar". A regular expression may have multiple capturing groups. In results, matches to capturing groups typically in an array whose members are in the same order as the left parentheses in the capturing group. This is usually just the order of the capturing groups themselves. This becomes important when capturing groups are nested. Matches are accessed using the index of the result's elements (`[1], …, [n]`) or from the predefined `RegExp` object's properties (`$1, …, $9`). Capturing groups have a performance penalty. If you don't need the matched substring to be recalled, prefer non-capturing parentheses (see below). `String.prototype.match()` won't return groups if the `/.../g` flag is set. However, you can still use `String.prototype.matchAll()` to get all matches. |
| `(?<Name>x)` | **Named capturing group:** Matches "x" and stores it on the groups property of the returned matches under the name specified by `<Name>`. The angle brackets (`<` and `>`) are required for group name. For example, to extract the United States area code from a phone number, we could use `/\((?<area>\d\d\d)\)/`. The resulting number would appear under `matches.groups.area`. |
| `(?:*x*)` | **Non-capturing group:** Matches "x" but does not remember the match. The matched substring cannot be recalled from the resulting array's elements (`[1], …, [n]`) or from the predefined `RegExp` object's properties (`$1, …, $9`). |
| `(?*flags*:*x*)`, `(?*flags*-*flags*:*x*)` | **Modifier:** Enables or disables the specified flags only to the enclosed pattern. Only the `i`, `m`, and `s` flags can be used in a modifier. |
| `\*n*` | **Backreference:** Where "n" is a positive integer. Matches the same substring matched by the nth capturing group in the regular expression (counting left parentheses). For example, `/apple(,)\sorange\1/` matches "apple, orange," in "apple, orange, cherry, peach". |
| `\k<Name>` | **Named backreference:** A back reference to the last substring matching the **Named capture group** specified by `<Name>`. For example, `/(?<title>\w+), yes \k<title>/` matches "Sir, yes Sir" in "Do you copy? Sir, yes Sir!". **Note:** `\k` is used literally here to indicate the beginning of a back reference to a Named capture group. |

## Quantifiers

Quantifiers indicate numbers of characters or expressions to match.

> **Note:**
> In the following, _item_ refers not only to singular characters, but also includes character classes and groups and backreferences.

| Characters | Meaning |
| --- | --- |
| `*x**` | Matches the preceding item "x" 0 or more times. For example, `/bo*/` matches "boooo" in "A ghost booooed" and "b" in "A bird warbled", but nothing in "A goat grunted". |
| `*x*+` | Matches the preceding item "x" 1 or more times. Equivalent to `{1,}`. For example, `/a+/` matches the "a" in "candy" and all the "a"'s in "caaaaaaandy". |
| `*x*?` | Matches the preceding item "x" 0 or 1 times. For example, `/e?le?/` matches the "el" in "angel" and the "le" in "angle." If used immediately after any of the quantifiers `*`, `+`, `?`, or `{}`, makes the quantifier non-greedy (matching the minimum number of times), as opposed to the default, which is greedy (matching the maximum number of times). |
| `*x*{*n*}` | Where "n" is a non-negative integer, matches exactly "n" occurrences of the preceding item "x". For example, `/a{2}/` doesn't match the "a" in "candy", but it matches all of the "a"'s in "caandy", and the first two "a"'s in "caaandy". |
| `*x*{*n*,}` | Where "n" is a non-negative integer, matches at least "n" occurrences of the preceding item "x". For example, `/a{2,}/` doesn't match the "a" in "candy", but matches all of the a's in "caandy" and in "caaaaaaandy". |
| `*x*{*n*,*m*}` | Where "n" and "m" are non-negative integers and `m >= n`, matches at least "n" and at most "m" occurrences of the preceding item "x". For example, `/a{1,3}/` matches nothing in "cndy", the "a" in "candy", the two "a"'s in "caandy", and the first three "a"'s in "caaaaaaandy". Notice that when matching "caaaaaaandy", the match is "aaa", even though the original string had more "a"s in it. |
| `*x**?` `*x*+?` `*x*??` `*x*{n}?` `*x*{n,}?` `*x*{n,m}?` | By default quantifiers like `*` and `+` are "greedy", meaning that they try to match as many times as possible. The `?` character after the quantifier makes the quantifier "non-greedy": meaning that it will stop as soon as it finds the minimum number of matches. For example, given a string like "some <foo> <bar> new </bar> </foo> thing": `/<.*>/` will match "<foo> <bar> new </bar> </foo>" `/<.*?>/` will match "<foo>" **Note:** Adding `?` after `{n}` is syntactically valid but practically useless. Since `{n}` always matches exactly n times, `x{n}?` behaves the same as `x{n}`. |
