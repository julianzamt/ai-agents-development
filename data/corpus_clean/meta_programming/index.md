---
title: Meta programming
slug: Web/JavaScript/Guide/Meta_programming
---

The `Proxy` and `Reflect` objects allow you to intercept and define custom behavior for fundamental language operations (e.g., property lookup, assignment, enumeration, function invocation, etc.). With the help of these two objects you are able to program at the meta level of JavaScript.

## Proxies

`Proxy` objects allow you to intercept certain operations and to implement custom behaviors.

For example, getting a property on an object:

```js
const handler = {
  get(target, name) {
    return name in target ? target[name] : 42;
  },
};

const p = new Proxy({}, handler);
p.a = 1;
console.log(p.a, p.b); // 1, 42
```

The `Proxy` object defines a `target` (an empty object here) and a `handler` object, in which a `get` _trap_ is implemented. Here, an object that is proxied will not return `undefined` when getting undefined properties, but will instead return the number `42`.

Additional examples are available on the `Proxy` reference page.

### Terminology

The following terms are used when talking about the functionality of proxies.

- `handler`
  - : Placeholder object which contains traps.
- traps
  - : The methods that provide property access. (This is analogous to the concept of _traps_ in operating systems.)
- target
  - : Object which the proxy virtualizes. It is often used as storage backend for the proxy. Invariants (semantics that remain unchanged) regarding object non-extensibility or non-configurable properties are verified against the target.
- invariants
  - : Semantics that remain unchanged when implementing custom operations are called _invariants_. If you violate the invariants of a handler, a `TypeError` will be thrown.

## Handlers and traps

The following table summarizes the available traps available to `Proxy` objects. See the reference pages for detailed explanations and examples.

| Handler / trap | Interceptions |
| --- | --- |
| `handler.getPrototypeOf()` | `Object.getPrototypeOf()` `Reflect.getPrototypeOf()` `__proto__` `Object.prototype.isPrototypeOf()` `instanceof` |
| `handler.setPrototypeOf()` | `Object.setPrototypeOf()` `Reflect.setPrototypeOf()` |
| `handler.isExtensible()` | `Object.isExtensible()` `Reflect.isExtensible()` |
| `handler.preventExtensions()` | `Object.preventExtensions()` `Reflect.preventExtensions()` |
| `handler.getOwnPropertyDescriptor()` | `Object.getOwnPropertyDescriptor()` `Reflect.getOwnPropertyDescriptor()` |
| `handler.defineProperty()` | `Object.defineProperty()` `Reflect.defineProperty()` |
| `handler.has()` | Property query `foo in proxy` Inherited property query `foo in Object.create(proxy)` `Reflect.has()` |
| `handler.get()` | Property access `proxy[foo]` `proxy.bar` Inherited property access `Object.create(proxy)[foo]` `Reflect.get()` |
| `handler.set()` | Property assignment `proxy[foo] = bar` `proxy.foo = bar` Inherited property assignment `Object.create(proxy)[foo] = bar` `Reflect.set()` |
| `handler.deleteProperty()` | Property deletion `delete proxy[foo]` `delete proxy.foo` `Reflect.deleteProperty()` |
| `handler.ownKeys()` | `Object.getOwnPropertyNames()` `Object.getOwnPropertySymbols()` `Object.keys()` `Reflect.ownKeys()` |
| `handler.apply()` | `proxy(..args)` `Function.prototype.apply()` and `Function.prototype.call()` `Reflect.apply()` |
| `handler.construct()` | `new proxy(...args)` `Reflect.construct()` |

## Revocable `Proxy`

The `Proxy.revocable()` method is used to create a revocable `Proxy` object. This means that the proxy can be revoked via the function `revoke` and switches the proxy off.

Afterwards, any operation on the proxy leads to a `TypeError`.

```js
const revocable = Proxy.revocable(
  {},
  {
    get(target, name) {
      return `[[${name}]]`;
    },
  },
);
const proxy = revocable.proxy;
console.log(proxy.foo); // "[[foo]]"

revocable.revoke();

console.log(proxy.foo); // TypeError: Cannot perform 'get' on a proxy that has been revoked
proxy.foo = 1; // TypeError: Cannot perform 'set' on a proxy that has been revoked
delete proxy.foo; // TypeError: Cannot perform 'deleteProperty' on a proxy that has been revoked
console.log(typeof proxy); // "object", typeof doesn't trigger any trap
```

## Reflection

`Reflect` is a built-in object that provides methods for interceptable JavaScript operations. The methods are the same as those of the proxy handler's.

`Reflect` is not a function object.

`Reflect` helps with forwarding default operations from the handler to the `target`.

With `Reflect.has()` for example, you get the `in` operator as a function:

```js
Reflect.has(Object, "assign"); // true
```

### A better apply() function

Before `Reflect`, you typically use the `Function.prototype.apply()` method to call a function with a given `this` value and `arguments` provided as an array (or an array-like object).

```js
Function.prototype.apply.call(Math.floor, undefined, [1.75]);
```

With `Reflect.apply` this becomes less verbose and easier to understand:

```js
Reflect.apply(Math.floor, undefined, [1.75]);
// 1

Reflect.apply(String.fromCharCode, undefined, [104, 101, 108, 108, 111]);
// "hello"

Reflect.apply(RegExp.prototype.exec, /ab/, ["confabulation"]).index;
// 4

Reflect.apply("".charAt, "ponies", [3]);
// "i"
```

### Checking if property definition has been successful

With `Object.defineProperty`, which returns an object if successful, or throws a `TypeError` otherwise, you would use a `try...catch` block to catch any error that occurred while defining a property. Because `Reflect.defineProperty()` returns a Boolean success status, you can just use an `if...else` block here:

```js
if (Reflect.defineProperty(target, property, attributes)) {
  // success
} else {
  // failure
}
```
