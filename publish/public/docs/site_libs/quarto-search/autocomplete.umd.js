/*! @algolia/autocomplete-js 1.11.1 | MIT License | © Algolia, Inc. and contributors | https://github.com/algolia/autocomplete */
!(function (e, t) {
  "object" == typeof exports && "undefined" != typeof module
    ? t(exports)
    : "function" == typeof define && define.amd
    ? define(["exports"], t)
    : t(
        ((e = "undefined" != typeof globalThis ? globalThis : e || self)[
          "@algolia/autocomplete-js"
        ] = {})
      );
})(this, function (e) {
  "use strict";
  function t(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function n(e) {
    for (var n = 1; n < arguments.length; n++) {
      var r = null != arguments[n] ? arguments[n] : {};
      n % 2
        ? t(Object(r), !0).forEach(function (t) {
            o(e, t, r[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r))
        : t(Object(r)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t));
          });
    }
    return e;
  }
  function r(e) {
    return (
      (r =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      r(e)
    );
  }
  function o(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" != typeof e || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" != typeof r) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" == typeof t ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function i() {
    return (
      (i = Object.assign
        ? Object.assign.bind()
        : function (e) {
            for (var t = 1; t < arguments.length; t++) {
              var n = arguments[t];
              for (var r in n)
                Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
            }
            return e;
          }),
      i.apply(this, arguments)
    );
  }
  function u(e, t) {
    if (null == e) return {};
    var n,
      r,
      o = (function (e, t) {
        if (null == e) return {};
        var n,
          r,
          o = {},
          i = Object.keys(e);
        for (r = 0; r < i.length; r++)
          (n = i[r]), t.indexOf(n) >= 0 || (o[n] = e[n]);
        return o;
      })(e, t);
    if (Object.getOwnPropertySymbols) {
      var i = Object.getOwnPropertySymbols(e);
      for (r = 0; r < i.length; r++)
        (n = i[r]),
          t.indexOf(n) >= 0 ||
            (Object.prototype.propertyIsEnumerable.call(e, n) && (o[n] = e[n]));
    }
    return o;
  }
  function a(e, t) {
    return (
      (function (e) {
        if (Array.isArray(e)) return e;
      })(e) ||
      (function (e, t) {
        var n =
          null == e
            ? null
            : ("undefined" != typeof Symbol && e[Symbol.iterator]) ||
              e["@@iterator"];
        if (null != n) {
          var r,
            o,
            i,
            u,
            a = [],
            l = !0,
            c = !1;
          try {
            if (((i = (n = n.call(e)).next), 0 === t)) {
              if (Object(n) !== n) return;
              l = !1;
            } else
              for (
                ;
                !(l = (r = i.call(n)).done) &&
                (a.push(r.value), a.length !== t);
                l = !0
              );
          } catch (e) {
            (c = !0), (o = e);
          } finally {
            try {
              if (!l && null != n.return && ((u = n.return()), Object(u) !== u))
                return;
            } finally {
              if (c) throw o;
            }
          }
          return a;
        }
      })(e, t) ||
      c(e, t) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function l(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return s(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      c(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function c(e, t) {
    if (e) {
      if ("string" == typeof e) return s(e, t);
      var n = Object.prototype.toString.call(e).slice(8, -1);
      return (
        "Object" === n && e.constructor && (n = e.constructor.name),
        "Map" === n || "Set" === n
          ? Array.from(e)
          : "Arguments" === n ||
            /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
          ? s(e, t)
          : void 0
      );
    }
  }
  function s(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function f(e) {
    return { current: e };
  }
  function p(e, t) {
    var n = void 0;
    return function () {
      for (var r = arguments.length, o = new Array(r), i = 0; i < r; i++)
        o[i] = arguments[i];
      n && clearTimeout(n),
        (n = setTimeout(function () {
          return e.apply(void 0, o);
        }, t));
    };
  }
  function m(e) {
    return e.reduce(function (e, t) {
      return e.concat(t);
    }, []);
  }
  var v = 0;
  function d() {
    return "autocomplete-".concat(v++);
  }
  function y(e, t) {
    return t.reduce(function (e, t) {
      return e && e[t];
    }, e);
  }
  function b(e) {
    return 0 === e.collections.length
      ? 0
      : e.collections.reduce(function (e, t) {
          return e + t.items.length;
        }, 0);
  }
  function g(e) {
    return e !== Object(e);
  }
  function h(e, t) {
    if (e === t) return !0;
    if (g(e) || g(t) || "function" == typeof e || "function" == typeof t)
      return e === t;
    if (Object.keys(e).length !== Object.keys(t).length) return !1;
    for (var n = 0, r = Object.keys(e); n < r.length; n++) {
      var o = r[n];
      if (!(o in t)) return !1;
      if (!h(e[o], t[o])) return !1;
    }
    return !0;
  }
  var O = function () {};
  var _ = "1.11.1",
    S = [{ segment: "autocomplete-core", version: _ }];
  function j(e) {
    var t = e.item,
      n = e.items,
      r = void 0 === n ? [] : n;
    return {
      index: t.__autocomplete_indexName,
      items: [t],
      positions: [
        1 +
          r.findIndex(function (e) {
            return e.objectID === t.objectID;
          }),
      ],
      queryID: t.__autocomplete_queryID,
      algoliaSource: ["autocomplete"],
    };
  }
  function P(e, t) {
    return (
      (function (e) {
        if (Array.isArray(e)) return e;
      })(e) ||
      (function (e, t) {
        var n =
          null == e
            ? null
            : ("undefined" != typeof Symbol && e[Symbol.iterator]) ||
              e["@@iterator"];
        if (null != n) {
          var r,
            o,
            i,
            u,
            a = [],
            l = !0,
            c = !1;
          try {
            if (((i = (n = n.call(e)).next), 0 === t)) {
              if (Object(n) !== n) return;
              l = !1;
            } else
              for (
                ;
                !(l = (r = i.call(n)).done) &&
                (a.push(r.value), a.length !== t);
                l = !0
              );
          } catch (e) {
            (c = !0), (o = e);
          } finally {
            try {
              if (!l && null != n.return && ((u = n.return()), Object(u) !== u))
                return;
            } finally {
              if (c) throw o;
            }
          }
          return a;
        }
      })(e, t) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return w(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return w(e, t);
      })(e, t) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function w(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  var I = ["items"],
    A = ["items"];
  function E(e) {
    return (
      (E =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      E(e)
    );
  }
  function D(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return C(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return C(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return C(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function C(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function k(e, t) {
    if (null == e) return {};
    var n,
      r,
      o = (function (e, t) {
        if (null == e) return {};
        var n,
          r,
          o = {},
          i = Object.keys(e);
        for (r = 0; r < i.length; r++)
          (n = i[r]), t.indexOf(n) >= 0 || (o[n] = e[n]);
        return o;
      })(e, t);
    if (Object.getOwnPropertySymbols) {
      var i = Object.getOwnPropertySymbols(e);
      for (r = 0; r < i.length; r++)
        (n = i[r]),
          t.indexOf(n) >= 0 ||
            (Object.prototype.propertyIsEnumerable.call(e, n) && (o[n] = e[n]));
    }
    return o;
  }
  function x(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function N(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? x(Object(n), !0).forEach(function (t) {
            T(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : x(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function T(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== E(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== E(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === E(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function q(e) {
    for (
      var t =
          arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : 20,
        n = [],
        r = 0;
      r < e.objectIDs.length;
      r += t
    )
      n.push(N(N({}, e), {}, { objectIDs: e.objectIDs.slice(r, r + t) }));
    return n;
  }
  function B(e) {
    return e.map(function (e) {
      var t = e.items,
        n = k(e, I);
      return N(
        N({}, n),
        {},
        {
          objectIDs:
            (null == t
              ? void 0
              : t.map(function (e) {
                  return e.objectID;
                })) || n.objectIDs,
        }
      );
    });
  }
  function R(e) {
    var t,
      n,
      r,
      o =
        ((t = P((e.version || "").split(".").map(Number), 2)),
        (n = t[0]),
        (r = t[1]),
        n >= 3 || (2 === n && r >= 4) || (1 === n && r >= 10));
    function i(t, n, r) {
      if (o && void 0 !== r) {
        var i = r[0].__autocomplete_algoliaCredentials,
          u = {
            "X-Algolia-Application-Id": i.appId,
            "X-Algolia-API-Key": i.apiKey,
          };
        e.apply(void 0, [t].concat(D(n), [{ headers: u }]));
      } else e.apply(void 0, [t].concat(D(n)));
    }
    return {
      init: function (t, n) {
        e("init", { appId: t, apiKey: n });
      },
      setUserToken: function (t) {
        e("setUserToken", t);
      },
      clickedObjectIDsAfterSearch: function () {
        for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
          t[n] = arguments[n];
        t.length > 0 && i("clickedObjectIDsAfterSearch", B(t), t[0].items);
      },
      clickedObjectIDs: function () {
        for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
          t[n] = arguments[n];
        t.length > 0 && i("clickedObjectIDs", B(t), t[0].items);
      },
      clickedFilters: function () {
        for (var t = arguments.length, n = new Array(t), r = 0; r < t; r++)
          n[r] = arguments[r];
        n.length > 0 && e.apply(void 0, ["clickedFilters"].concat(n));
      },
      convertedObjectIDsAfterSearch: function () {
        for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
          t[n] = arguments[n];
        t.length > 0 && i("convertedObjectIDsAfterSearch", B(t), t[0].items);
      },
      convertedObjectIDs: function () {
        for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
          t[n] = arguments[n];
        t.length > 0 && i("convertedObjectIDs", B(t), t[0].items);
      },
      convertedFilters: function () {
        for (var t = arguments.length, n = new Array(t), r = 0; r < t; r++)
          n[r] = arguments[r];
        n.length > 0 && e.apply(void 0, ["convertedFilters"].concat(n));
      },
      viewedObjectIDs: function () {
        for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
          t[n] = arguments[n];
        t.length > 0 &&
          t
            .reduce(function (e, t) {
              var n = t.items,
                r = k(t, A);
              return [].concat(
                D(e),
                D(
                  q(
                    N(
                      N({}, r),
                      {},
                      {
                        objectIDs:
                          (null == n
                            ? void 0
                            : n.map(function (e) {
                                return e.objectID;
                              })) || r.objectIDs,
                      }
                    )
                  ).map(function (e) {
                    return { items: n, payload: e };
                  })
                )
              );
            }, [])
            .forEach(function (e) {
              var t = e.items;
              return i("viewedObjectIDs", [e.payload], t);
            });
      },
      viewedFilters: function () {
        for (var t = arguments.length, n = new Array(t), r = 0; r < t; r++)
          n[r] = arguments[r];
        n.length > 0 && e.apply(void 0, ["viewedFilters"].concat(n));
      },
    };
  }
  function F(e) {
    var t = e.items.reduce(function (e, t) {
      var n;
      return (
        (e[t.__autocomplete_indexName] = (
          null !== (n = e[t.__autocomplete_indexName]) && void 0 !== n ? n : []
        ).concat(t)),
        e
      );
    }, {});
    return Object.keys(t).map(function (e) {
      return { index: e, items: t[e], algoliaSource: ["autocomplete"] };
    });
  }
  function L(e) {
    return e.objectID && e.__autocomplete_indexName && e.__autocomplete_queryID;
  }
  function U(e) {
    return (
      (U =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      U(e)
    );
  }
  function M(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return H(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return H(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return H(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function H(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function V(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function W(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? V(Object(n), !0).forEach(function (t) {
            K(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : V(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function K(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== U(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== U(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === U(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  var Q = "2.6.0",
    $ = "https://cdn.jsdelivr.net/npm/search-insights@".concat(
      Q,
      "/dist/search-insights.min.js"
    ),
    z = p(function (e) {
      var t = e.onItemsChange,
        n = e.items,
        r = e.insights,
        o = e.state;
      t({
        insights: r,
        insightsEvents: F({ items: n }).map(function (e) {
          return W({ eventName: "Items Viewed" }, e);
        }),
        state: o,
      });
    }, 400);
  function G(e) {
    var t = (function (e) {
        return W(
          {
            onItemsChange: function (e) {
              var t = e.insights,
                n = e.insightsEvents;
              t.viewedObjectIDs.apply(
                t,
                M(
                  n.map(function (e) {
                    return W(
                      W({}, e),
                      {},
                      {
                        algoliaSource: [].concat(M(e.algoliaSource || []), [
                          "autocomplete-internal",
                        ]),
                      }
                    );
                  })
                )
              );
            },
            onSelect: function (e) {
              var t = e.insights,
                n = e.insightsEvents;
              t.clickedObjectIDsAfterSearch.apply(
                t,
                M(
                  n.map(function (e) {
                    return W(
                      W({}, e),
                      {},
                      {
                        algoliaSource: [].concat(M(e.algoliaSource || []), [
                          "autocomplete-internal",
                        ]),
                      }
                    );
                  })
                )
              );
            },
            onActive: O,
          },
          e
        );
      })(e),
      n = t.insightsClient,
      r = t.onItemsChange,
      o = t.onSelect,
      i = t.onActive,
      u = n;
    n ||
      (function (e) {
        if ("undefined" != typeof window) e({ window: window });
      })(function (e) {
        var t = e.window,
          n = t.AlgoliaAnalyticsObject || "aa";
        "string" == typeof n && (u = t[n]),
          u ||
            ((t.AlgoliaAnalyticsObject = n),
            t[n] ||
              (t[n] = function () {
                t[n].queue || (t[n].queue = []);
                for (
                  var e = arguments.length, r = new Array(e), o = 0;
                  o < e;
                  o++
                )
                  r[o] = arguments[o];
                t[n].queue.push(r);
              }),
            (t[n].version = Q),
            (u = t[n]),
            (function (e) {
              var t =
                "[Autocomplete]: Could not load search-insights.js. Please load it manually following https://alg.li/insights-autocomplete";
              try {
                var n = e.document.createElement("script");
                (n.async = !0),
                  (n.src = $),
                  (n.onerror = function () {
                    console.error(t);
                  }),
                  document.body.appendChild(n);
              } catch (e) {
                console.error(t);
              }
            })(t));
      });
    var a = R(u),
      l = f([]),
      c = p(function (e) {
        var t = e.state;
        if (t.isOpen) {
          var n = t.collections
            .reduce(function (e, t) {
              return [].concat(M(e), M(t.items));
            }, [])
            .filter(L);
          h(
            l.current.map(function (e) {
              return e.objectID;
            }),
            n.map(function (e) {
              return e.objectID;
            })
          ) ||
            ((l.current = n),
            n.length > 0 &&
              z({ onItemsChange: r, items: n, insights: a, state: t }));
        }
      }, 0);
    return {
      name: "aa.algoliaInsightsPlugin",
      subscribe: function (e) {
        var t = e.setContext,
          n = e.onSelect,
          r = e.onActive;
        function l(e) {
          t({
            algoliaInsightsPlugin: {
              __algoliaSearchParameters: W(
                { clickAnalytics: !0 },
                e ? { userToken: e } : {}
              ),
              insights: a,
            },
          });
        }
        u("addAlgoliaAgent", "insights-plugin"),
          l(),
          u("onUserTokenChange", l),
          u("getUserToken", null, function (e, t) {
            l(t);
          }),
          n(function (e) {
            var t = e.item,
              n = e.state,
              r = e.event,
              i = e.source;
            L(t) &&
              o({
                state: n,
                event: r,
                insights: a,
                item: t,
                insightsEvents: [
                  W(
                    { eventName: "Item Selected" },
                    j({ item: t, items: i.getItems().filter(L) })
                  ),
                ],
              });
          }),
          r(function (e) {
            var t = e.item,
              n = e.source,
              r = e.state,
              o = e.event;
            L(t) &&
              i({
                state: r,
                event: o,
                insights: a,
                item: t,
                insightsEvents: [
                  W(
                    { eventName: "Item Active" },
                    j({ item: t, items: n.getItems().filter(L) })
                  ),
                ],
              });
          });
      },
      onStateChange: function (e) {
        var t = e.state;
        c({ state: t });
      },
      __autocomplete_pluginOptions: e,
    };
  }
  function J(e, t) {
    var n = t;
    return {
      then: function (t, r) {
        return J(e.then(Y(t, n, e), Y(r, n, e)), n);
      },
      catch: function (t) {
        return J(e.catch(Y(t, n, e)), n);
      },
      finally: function (t) {
        return (
          t && n.onCancelList.push(t),
          J(
            e.finally(
              Y(
                t &&
                  function () {
                    return (n.onCancelList = []), t();
                  },
                n,
                e
              )
            ),
            n
          )
        );
      },
      cancel: function () {
        n.isCanceled = !0;
        var e = n.onCancelList;
        (n.onCancelList = []),
          e.forEach(function (e) {
            e();
          });
      },
      isCanceled: function () {
        return !0 === n.isCanceled;
      },
    };
  }
  function X(e) {
    return J(e, { isCanceled: !1, onCancelList: [] });
  }
  function Y(e, t, n) {
    return e
      ? function (n) {
          return t.isCanceled ? n : e(n);
        }
      : n;
  }
  function Z(e, t, n, r) {
    if (!n) return null;
    if (e < 0 && (null === t || (null !== r && 0 === t))) return n + e;
    var o = (null === t ? -1 : t) + e;
    return o <= -1 || o >= n ? (null === r ? null : 0) : o;
  }
  function ee(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function te(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? ee(Object(n), !0).forEach(function (t) {
            ne(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : ee(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function ne(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== re(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== re(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === re(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function re(e) {
    return (
      (re =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      re(e)
    );
  }
  function oe(e) {
    var t = (function (e) {
      var t = e.collections
        .map(function (e) {
          return e.items.length;
        })
        .reduce(function (e, t, n) {
          var r = (e[n - 1] || 0) + t;
          return e.push(r), e;
        }, [])
        .reduce(function (t, n) {
          return n <= e.activeItemId ? t + 1 : t;
        }, 0);
      return e.collections[t];
    })(e);
    if (!t) return null;
    var n =
        t.items[
          (function (e) {
            for (
              var t = e.state, n = e.collection, r = !1, o = 0, i = 0;
              !1 === r;

            ) {
              var u = t.collections[o];
              if (u === n) {
                r = !0;
                break;
              }
              (i += u.items.length), o++;
            }
            return t.activeItemId - i;
          })({ state: e, collection: t })
        ],
      r = t.source;
    return {
      item: n,
      itemInputValue: r.getItemInputValue({ item: n, state: e }),
      itemUrl: r.getItemUrl({ item: n, state: e }),
      source: r,
    };
  }
  function ie(e, t, n) {
    return [e, null == n ? void 0 : n.sourceId, t]
      .filter(Boolean)
      .join("-")
      .replace(/\s/g, "");
  }
  var ue = /((gt|sm)-|galaxy nexus)|samsung[- ]|samsungbrowser/i;
  function ae(e) {
    return (
      (ae =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      ae(e)
    );
  }
  function le(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function ce(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== ae(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== ae(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === ae(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function se(e, t, n) {
    var r,
      o = t.initialState;
    return {
      getState: function () {
        return o;
      },
      dispatch: function (r, i) {
        var u = (function (e) {
          for (var t = 1; t < arguments.length; t++) {
            var n = null != arguments[t] ? arguments[t] : {};
            t % 2
              ? le(Object(n), !0).forEach(function (t) {
                  ce(e, t, n[t]);
                })
              : Object.getOwnPropertyDescriptors
              ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
              : le(Object(n)).forEach(function (t) {
                  Object.defineProperty(
                    e,
                    t,
                    Object.getOwnPropertyDescriptor(n, t)
                  );
                });
          }
          return e;
        })({}, o);
        (o = e(o, { type: r, props: t, payload: i })),
          n({ state: o, prevState: u });
      },
      pendingRequests:
        ((r = []),
        {
          add: function (e) {
            return (
              r.push(e),
              e.finally(function () {
                r = r.filter(function (t) {
                  return t !== e;
                });
              })
            );
          },
          cancelAll: function () {
            r.forEach(function (e) {
              return e.cancel();
            });
          },
          isEmpty: function () {
            return 0 === r.length;
          },
        }),
    };
  }
  function fe(e) {
    return (
      (fe =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      fe(e)
    );
  }
  function pe(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function me(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? pe(Object(n), !0).forEach(function (t) {
            ve(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : pe(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function ve(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== fe(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== fe(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === fe(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function de(e) {
    return (
      (de =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      de(e)
    );
  }
  function ye(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return be(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return be(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return be(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function be(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function ge(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function he(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? ge(Object(n), !0).forEach(function (t) {
            Oe(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : ge(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function Oe(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== de(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== de(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === de(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function _e(e, t) {
    var n,
      r = "undefined" != typeof window ? window : {},
      o = e.plugins || [];
    return he(
      he(
        {
          debug: !1,
          openOnFocus: !1,
          enterKeyHint: void 0,
          placeholder: "",
          autoFocus: !1,
          defaultActiveItemId: null,
          stallThreshold: 300,
          insights: !1,
          environment: r,
          shouldPanelOpen: function (e) {
            return b(e.state) > 0;
          },
          reshape: function (e) {
            return e.sources;
          },
        },
        e
      ),
      {},
      {
        id: null !== (n = e.id) && void 0 !== n ? n : d(),
        plugins: o,
        initialState: he(
          {
            activeItemId: null,
            query: "",
            completion: null,
            collections: [],
            isOpen: !1,
            status: "idle",
            context: {},
          },
          e.initialState
        ),
        onStateChange: function (t) {
          var n;
          null === (n = e.onStateChange) || void 0 === n || n.call(e, t),
            o.forEach(function (e) {
              var n;
              return null === (n = e.onStateChange) || void 0 === n
                ? void 0
                : n.call(e, t);
            });
        },
        onSubmit: function (t) {
          var n;
          null === (n = e.onSubmit) || void 0 === n || n.call(e, t),
            o.forEach(function (e) {
              var n;
              return null === (n = e.onSubmit) || void 0 === n
                ? void 0
                : n.call(e, t);
            });
        },
        onReset: function (t) {
          var n;
          null === (n = e.onReset) || void 0 === n || n.call(e, t),
            o.forEach(function (e) {
              var n;
              return null === (n = e.onReset) || void 0 === n
                ? void 0
                : n.call(e, t);
            });
        },
        getSources: function (n) {
          return Promise.all(
            []
              .concat(
                ye(
                  o.map(function (e) {
                    return e.getSources;
                  })
                ),
                [e.getSources]
              )
              .filter(Boolean)
              .map(function (e) {
                return (function (e, t) {
                  var n = [];
                  return Promise.resolve(e(t)).then(function (e) {
                    return Promise.all(
                      e
                        .filter(function (e) {
                          return Boolean(e);
                        })
                        .map(function (e) {
                          if ((e.sourceId, n.includes(e.sourceId)))
                            throw new Error(
                              "[Autocomplete] The `sourceId` ".concat(
                                JSON.stringify(e.sourceId),
                                " is not unique."
                              )
                            );
                          n.push(e.sourceId);
                          var t = {
                            getItemInputValue: function (e) {
                              return e.state.query;
                            },
                            getItemUrl: function () {},
                            onSelect: function (e) {
                              (0, e.setIsOpen)(!1);
                            },
                            onActive: O,
                            onResolve: O,
                          };
                          Object.keys(t).forEach(function (e) {
                            t[e].__default = !0;
                          });
                          var r = te(te({}, t), e);
                          return Promise.resolve(r);
                        })
                    );
                  });
                })(e, n);
              })
          )
            .then(function (e) {
              return m(e);
            })
            .then(function (e) {
              return e.map(function (e) {
                return he(
                  he({}, e),
                  {},
                  {
                    onSelect: function (n) {
                      e.onSelect(n),
                        t.forEach(function (e) {
                          var t;
                          return null === (t = e.onSelect) || void 0 === t
                            ? void 0
                            : t.call(e, n);
                        });
                    },
                    onActive: function (n) {
                      e.onActive(n),
                        t.forEach(function (e) {
                          var t;
                          return null === (t = e.onActive) || void 0 === t
                            ? void 0
                            : t.call(e, n);
                        });
                    },
                    onResolve: function (n) {
                      e.onResolve(n),
                        t.forEach(function (e) {
                          var t;
                          return null === (t = e.onResolve) || void 0 === t
                            ? void 0
                            : t.call(e, n);
                        });
                    },
                  }
                );
              });
            });
        },
        navigator: he(
          {
            navigate: function (e) {
              var t = e.itemUrl;
              r.location.assign(t);
            },
            navigateNewTab: function (e) {
              var t = e.itemUrl,
                n = r.open(t, "_blank", "noopener");
              null == n || n.focus();
            },
            navigateNewWindow: function (e) {
              var t = e.itemUrl;
              r.open(t, "_blank", "noopener");
            },
          },
          e.navigator
        ),
      }
    );
  }
  function Se(e) {
    return (
      (Se =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      Se(e)
    );
  }
  function je(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function Pe(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? je(Object(n), !0).forEach(function (t) {
            we(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : je(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function we(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== Se(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== Se(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === Se(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function Ie(e) {
    return (
      (Ie =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      Ie(e)
    );
  }
  function Ae(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function Ee(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? Ae(Object(n), !0).forEach(function (t) {
            De(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : Ae(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function De(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== Ie(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== Ie(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === Ie(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function Ce(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return ke(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return ke(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return ke(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function ke(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function xe(e) {
    return Boolean(e.execute);
  }
  function Ne(e, t, n) {
    if (((o = e), Boolean(null == o ? void 0 : o.execute))) {
      var r =
        "algolia" === e.requesterId
          ? Object.assign.apply(
              Object,
              [{}].concat(
                Ce(
                  Object.keys(n.context).map(function (e) {
                    var t;
                    return null === (t = n.context[e]) || void 0 === t
                      ? void 0
                      : t.__algoliaSearchParameters;
                  })
                )
              )
            )
          : {};
      return Ee(
        Ee({}, e),
        {},
        {
          requests: e.queries.map(function (n) {
            return {
              query:
                "algolia" === e.requesterId
                  ? Ee(Ee({}, n), {}, { params: Ee(Ee({}, r), n.params) })
                  : n,
              sourceId: t,
              transformResponse: e.transformResponse,
            };
          }),
        }
      );
    }
    var o;
    return { items: e, sourceId: t };
  }
  function Te(e) {
    var t = e
      .reduce(function (e, t) {
        if (!xe(t)) return e.push(t), e;
        var n = t.searchClient,
          r = t.execute,
          o = t.requesterId,
          i = t.requests,
          u = e.find(function (e) {
            return (
              xe(t) &&
              xe(e) &&
              e.searchClient === n &&
              Boolean(o) &&
              e.requesterId === o
            );
          });
        if (u) {
          var a;
          (a = u.items).push.apply(a, Ce(i));
        } else {
          var l = { execute: r, requesterId: o, items: i, searchClient: n };
          e.push(l);
        }
        return e;
      }, [])
      .map(function (e) {
        if (!xe(e)) return Promise.resolve(e);
        var t = e,
          n = t.execute,
          r = t.items;
        return n({ searchClient: t.searchClient, requests: r });
      });
    return Promise.all(t).then(function (e) {
      return m(e);
    });
  }
  function qe(e, t, n) {
    return t.map(function (t) {
      var r,
        o = e.filter(function (e) {
          return e.sourceId === t.sourceId;
        }),
        i = o.map(function (e) {
          return e.items;
        }),
        u = o[0].transformResponse,
        a = u
          ? u({
              results: (r = i),
              hits: r
                .map(function (e) {
                  return e.hits;
                })
                .filter(Boolean),
              facetHits: r
                .map(function (e) {
                  var t;
                  return null === (t = e.facetHits) || void 0 === t
                    ? void 0
                    : t.map(function (e) {
                        return {
                          label: e.value,
                          count: e.count,
                          _highlightResult: { label: { value: e.highlighted } },
                        };
                      });
                })
                .filter(Boolean),
            })
          : i;
      return (
        t.onResolve({ source: t, results: i, items: a, state: n.getState() }),
        a.every(Boolean),
        'The `getItems` function from source "'
          .concat(t.sourceId, '" must return an array of items but returned ')
          .concat(
            JSON.stringify(void 0),
            ".\n\nDid you forget to return items?\n\nSee: https://www.algolia.com/doc/ui-libraries/autocomplete/core-concepts/sources/#param-getitems"
          ),
        { source: t, items: a }
      );
    });
  }
  function Be(e) {
    return (
      (Be =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      Be(e)
    );
  }
  var Re = ["event", "nextState", "props", "query", "refresh", "store"];
  function Fe(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function Le(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? Fe(Object(n), !0).forEach(function (t) {
            Ue(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : Fe(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function Ue(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== Be(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== Be(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === Be(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function Me(e, t) {
    if (null == e) return {};
    var n,
      r,
      o = (function (e, t) {
        if (null == e) return {};
        var n,
          r,
          o = {},
          i = Object.keys(e);
        for (r = 0; r < i.length; r++)
          (n = i[r]), t.indexOf(n) >= 0 || (o[n] = e[n]);
        return o;
      })(e, t);
    if (Object.getOwnPropertySymbols) {
      var i = Object.getOwnPropertySymbols(e);
      for (r = 0; r < i.length; r++)
        (n = i[r]),
          t.indexOf(n) >= 0 ||
            (Object.prototype.propertyIsEnumerable.call(e, n) && (o[n] = e[n]));
    }
    return o;
  }
  var He,
    Ve,
    We,
    Ke = null,
    Qe =
      ((He = -1),
      (Ve = -1),
      (We = void 0),
      function (e) {
        var t = ++He;
        return Promise.resolve(e).then(function (e) {
          return We && t < Ve ? We : ((Ve = t), (We = e), e);
        });
      });
  function $e(e) {
    var t = e.event,
      n = e.nextState,
      r = void 0 === n ? {} : n,
      o = e.props,
      i = e.query,
      u = e.refresh,
      a = e.store,
      l = Me(e, Re);
    Ke && o.environment.clearTimeout(Ke);
    var c = l.setCollections,
      s = l.setIsOpen,
      f = l.setQuery,
      p = l.setActiveItemId,
      v = l.setStatus;
    if ((f(i), p(o.defaultActiveItemId), !i && !1 === o.openOnFocus)) {
      var d,
        y = a.getState().collections.map(function (e) {
          return Le(Le({}, e), {}, { items: [] });
        });
      v("idle"),
        c(y),
        s(
          null !== (d = r.isOpen) && void 0 !== d
            ? d
            : o.shouldPanelOpen({ state: a.getState() })
        );
      var b = X(
        Qe(y).then(function () {
          return Promise.resolve();
        })
      );
      return a.pendingRequests.add(b);
    }
    v("loading"),
      (Ke = o.environment.setTimeout(function () {
        v("stalled");
      }, o.stallThreshold));
    var g = X(
      Qe(
        o
          .getSources(Le({ query: i, refresh: u, state: a.getState() }, l))
          .then(function (e) {
            return Promise.all(
              e.map(function (e) {
                return Promise.resolve(
                  e.getItems(
                    Le({ query: i, refresh: u, state: a.getState() }, l)
                  )
                ).then(function (t) {
                  return Ne(t, e.sourceId, a.getState());
                });
              })
            )
              .then(Te)
              .then(function (t) {
                return qe(t, e, a);
              })
              .then(function (e) {
                return (function (e) {
                  var t = e.collections,
                    n = e.props,
                    r = e.state,
                    o = t.reduce(function (e, t) {
                      return Pe(
                        Pe({}, e),
                        {},
                        we(
                          {},
                          t.source.sourceId,
                          Pe(
                            Pe({}, t.source),
                            {},
                            {
                              getItems: function () {
                                return m(t.items);
                              },
                            }
                          )
                        )
                      );
                    }, {}),
                    i = n.plugins.reduce(
                      function (e, t) {
                        return t.reshape ? t.reshape(e) : e;
                      },
                      { sourcesBySourceId: o, state: r }
                    ).sourcesBySourceId;
                  return m(
                    n.reshape({
                      sourcesBySourceId: i,
                      sources: Object.values(i),
                      state: r,
                    })
                  )
                    .filter(Boolean)
                    .map(function (e) {
                      return { source: e, items: e.getItems() };
                    });
                })({ collections: e, props: o, state: a.getState() });
              });
          })
      )
    )
      .then(function (e) {
        var n;
        v("idle"), c(e);
        var f = o.shouldPanelOpen({ state: a.getState() });
        s(
          null !== (n = r.isOpen) && void 0 !== n
            ? n
            : (o.openOnFocus && !i && f) || f
        );
        var p = oe(a.getState());
        if (null !== a.getState().activeItemId && p) {
          var m = p.item,
            d = p.itemInputValue,
            y = p.itemUrl,
            b = p.source;
          b.onActive(
            Le(
              {
                event: t,
                item: m,
                itemInputValue: d,
                itemUrl: y,
                refresh: u,
                source: b,
                state: a.getState(),
              },
              l
            )
          );
        }
      })
      .finally(function () {
        v("idle"), Ke && o.environment.clearTimeout(Ke);
      });
    return a.pendingRequests.add(g);
  }
  function ze(e) {
    return (
      (ze =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      ze(e)
    );
  }
  var Ge = ["event", "props", "refresh", "store"];
  function Je(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function Xe(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? Je(Object(n), !0).forEach(function (t) {
            Ye(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : Je(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function Ye(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== ze(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== ze(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === ze(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function Ze(e, t) {
    if (null == e) return {};
    var n,
      r,
      o = (function (e, t) {
        if (null == e) return {};
        var n,
          r,
          o = {},
          i = Object.keys(e);
        for (r = 0; r < i.length; r++)
          (n = i[r]), t.indexOf(n) >= 0 || (o[n] = e[n]);
        return o;
      })(e, t);
    if (Object.getOwnPropertySymbols) {
      var i = Object.getOwnPropertySymbols(e);
      for (r = 0; r < i.length; r++)
        (n = i[r]),
          t.indexOf(n) >= 0 ||
            (Object.prototype.propertyIsEnumerable.call(e, n) && (o[n] = e[n]));
    }
    return o;
  }
  function et(e) {
    return (
      (et =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      et(e)
    );
  }
  var tt = ["props", "refresh", "store"],
    nt = ["inputElement", "formElement", "panelElement"],
    rt = ["inputElement"],
    ot = ["inputElement", "maxLength"],
    it = ["source"],
    ut = ["item", "source"];
  function at(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function lt(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? at(Object(n), !0).forEach(function (t) {
            ct(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : at(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function ct(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== et(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== et(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === et(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function st(e, t) {
    if (null == e) return {};
    var n,
      r,
      o = (function (e, t) {
        if (null == e) return {};
        var n,
          r,
          o = {},
          i = Object.keys(e);
        for (r = 0; r < i.length; r++)
          (n = i[r]), t.indexOf(n) >= 0 || (o[n] = e[n]);
        return o;
      })(e, t);
    if (Object.getOwnPropertySymbols) {
      var i = Object.getOwnPropertySymbols(e);
      for (r = 0; r < i.length; r++)
        (n = i[r]),
          t.indexOf(n) >= 0 ||
            (Object.prototype.propertyIsEnumerable.call(e, n) && (o[n] = e[n]));
    }
    return o;
  }
  function ft(e) {
    var t = e.props,
      n = e.refresh,
      r = e.store,
      o = st(e, tt);
    return {
      getEnvironmentProps: function (e) {
        var n = e.inputElement,
          o = e.formElement,
          i = e.panelElement;
        function u(e) {
          (!r.getState().isOpen && r.pendingRequests.isEmpty()) ||
            e.target === n ||
            (!1 ===
              [o, i].some(function (t) {
                return (n = t), (r = e.target), n === r || n.contains(r);
                var n, r;
              }) &&
              (r.dispatch("blur", null),
              t.debug || r.pendingRequests.cancelAll()));
        }
        return lt(
          {
            onTouchStart: u,
            onMouseDown: u,
            onTouchMove: function (e) {
              !1 !== r.getState().isOpen &&
                n === t.environment.document.activeElement &&
                e.target !== n &&
                n.blur();
            },
          },
          st(e, nt)
        );
      },
      getRootProps: function (e) {
        return lt(
          {
            role: "combobox",
            "aria-expanded": r.getState().isOpen,
            "aria-haspopup": "listbox",
            "aria-owns": r.getState().isOpen
              ? r
                  .getState()
                  .collections.map(function (e) {
                    var n = e.source;
                    return ie(t.id, "list", n);
                  })
                  .join(" ")
              : void 0,
            "aria-labelledby": ie(t.id, "label"),
          },
          e
        );
      },
      getFormProps: function (e) {
        return (
          e.inputElement,
          lt(
            {
              action: "",
              noValidate: !0,
              role: "search",
              onSubmit: function (i) {
                var u;
                i.preventDefault(),
                  t.onSubmit(
                    lt({ event: i, refresh: n, state: r.getState() }, o)
                  ),
                  r.dispatch("submit", null),
                  null === (u = e.inputElement) || void 0 === u || u.blur();
              },
              onReset: function (i) {
                var u;
                i.preventDefault(),
                  t.onReset(
                    lt({ event: i, refresh: n, state: r.getState() }, o)
                  ),
                  r.dispatch("reset", null),
                  null === (u = e.inputElement) || void 0 === u || u.focus();
              },
            },
            st(e, rt)
          )
        );
      },
      getLabelProps: function (e) {
        return lt({ htmlFor: ie(t.id, "input"), id: ie(t.id, "label") }, e);
      },
      getInputProps: function (e) {
        var i;
        function u(e) {
          (t.openOnFocus || Boolean(r.getState().query)) &&
            $e(
              lt(
                {
                  event: e,
                  props: t,
                  query: r.getState().completion || r.getState().query,
                  refresh: n,
                  store: r,
                },
                o
              )
            ),
            r.dispatch("focus", null);
        }
        var a = e || {};
        a.inputElement;
        var l = a.maxLength,
          c = void 0 === l ? 512 : l,
          s = st(a, ot),
          f = oe(r.getState()),
          p = (function (e) {
            return Boolean(e && e.match(ue));
          })(
            (null === (i = t.environment.navigator) || void 0 === i
              ? void 0
              : i.userAgent) || ""
          ),
          m =
            t.enterKeyHint || (null != f && f.itemUrl && !p ? "go" : "search");
        return lt(
          {
            "aria-autocomplete": "both",
            "aria-activedescendant":
              r.getState().isOpen && null !== r.getState().activeItemId
                ? ie(
                    t.id,
                    "item-".concat(r.getState().activeItemId),
                    null == f ? void 0 : f.source
                  )
                : void 0,
            "aria-controls": r.getState().isOpen
              ? r
                  .getState()
                  .collections.map(function (e) {
                    var n = e.source;
                    return ie(t.id, "list", n);
                  })
                  .join(" ")
              : void 0,
            "aria-labelledby": ie(t.id, "label"),
            value: r.getState().completion || r.getState().query,
            id: ie(t.id, "input"),
            autoComplete: "off",
            autoCorrect: "off",
            autoCapitalize: "off",
            enterKeyHint: m,
            spellCheck: "false",
            autoFocus: t.autoFocus,
            placeholder: t.placeholder,
            maxLength: c,
            type: "search",
            onChange: function (e) {
              $e(
                lt(
                  {
                    event: e,
                    props: t,
                    query: e.currentTarget.value.slice(0, c),
                    refresh: n,
                    store: r,
                  },
                  o
                )
              );
            },
            onKeyDown: function (e) {
              !(function (e) {
                var t = e.event,
                  n = e.props,
                  r = e.refresh,
                  o = e.store,
                  i = Ze(e, Ge);
                if ("ArrowUp" === t.key || "ArrowDown" === t.key) {
                  var u = function () {
                      var e = oe(o.getState()),
                        t = n.environment.document.getElementById(
                          ie(
                            n.id,
                            "item-".concat(o.getState().activeItemId),
                            null == e ? void 0 : e.source
                          )
                        );
                      t &&
                        (t.scrollIntoViewIfNeeded
                          ? t.scrollIntoViewIfNeeded(!1)
                          : t.scrollIntoView(!1));
                    },
                    a = function () {
                      var e = oe(o.getState());
                      if (null !== o.getState().activeItemId && e) {
                        var n = e.item,
                          u = e.itemInputValue,
                          a = e.itemUrl,
                          l = e.source;
                        l.onActive(
                          Xe(
                            {
                              event: t,
                              item: n,
                              itemInputValue: u,
                              itemUrl: a,
                              refresh: r,
                              source: l,
                              state: o.getState(),
                            },
                            i
                          )
                        );
                      }
                    };
                  t.preventDefault(),
                    !1 === o.getState().isOpen &&
                    (n.openOnFocus || Boolean(o.getState().query))
                      ? $e(
                          Xe(
                            {
                              event: t,
                              props: n,
                              query: o.getState().query,
                              refresh: r,
                              store: o,
                            },
                            i
                          )
                        ).then(function () {
                          o.dispatch(t.key, {
                            nextActiveItemId: n.defaultActiveItemId,
                          }),
                            a(),
                            setTimeout(u, 0);
                        })
                      : (o.dispatch(t.key, {}), a(), u());
                } else if ("Escape" === t.key)
                  t.preventDefault(),
                    o.dispatch(t.key, null),
                    o.pendingRequests.cancelAll();
                else if ("Tab" === t.key)
                  o.dispatch("blur", null), o.pendingRequests.cancelAll();
                else if ("Enter" === t.key) {
                  if (
                    null === o.getState().activeItemId ||
                    o.getState().collections.every(function (e) {
                      return 0 === e.items.length;
                    })
                  )
                    return void (n.debug || o.pendingRequests.cancelAll());
                  t.preventDefault();
                  var l = oe(o.getState()),
                    c = l.item,
                    s = l.itemInputValue,
                    f = l.itemUrl,
                    p = l.source;
                  if (t.metaKey || t.ctrlKey)
                    void 0 !== f &&
                      (p.onSelect(
                        Xe(
                          {
                            event: t,
                            item: c,
                            itemInputValue: s,
                            itemUrl: f,
                            refresh: r,
                            source: p,
                            state: o.getState(),
                          },
                          i
                        )
                      ),
                      n.navigator.navigateNewTab({
                        itemUrl: f,
                        item: c,
                        state: o.getState(),
                      }));
                  else if (t.shiftKey)
                    void 0 !== f &&
                      (p.onSelect(
                        Xe(
                          {
                            event: t,
                            item: c,
                            itemInputValue: s,
                            itemUrl: f,
                            refresh: r,
                            source: p,
                            state: o.getState(),
                          },
                          i
                        )
                      ),
                      n.navigator.navigateNewWindow({
                        itemUrl: f,
                        item: c,
                        state: o.getState(),
                      }));
                  else if (t.altKey);
                  else {
                    if (void 0 !== f)
                      return (
                        p.onSelect(
                          Xe(
                            {
                              event: t,
                              item: c,
                              itemInputValue: s,
                              itemUrl: f,
                              refresh: r,
                              source: p,
                              state: o.getState(),
                            },
                            i
                          )
                        ),
                        void n.navigator.navigate({
                          itemUrl: f,
                          item: c,
                          state: o.getState(),
                        })
                      );
                    $e(
                      Xe(
                        {
                          event: t,
                          nextState: { isOpen: !1 },
                          props: n,
                          query: s,
                          refresh: r,
                          store: o,
                        },
                        i
                      )
                    ).then(function () {
                      p.onSelect(
                        Xe(
                          {
                            event: t,
                            item: c,
                            itemInputValue: s,
                            itemUrl: f,
                            refresh: r,
                            source: p,
                            state: o.getState(),
                          },
                          i
                        )
                      );
                    });
                  }
                }
              })(lt({ event: e, props: t, refresh: n, store: r }, o));
            },
            onFocus: u,
            onBlur: O,
            onClick: function (n) {
              e.inputElement !== t.environment.document.activeElement ||
                r.getState().isOpen ||
                u(n);
            },
          },
          s
        );
      },
      getPanelProps: function (e) {
        return lt(
          {
            onMouseDown: function (e) {
              e.preventDefault();
            },
            onMouseLeave: function () {
              r.dispatch("mouseleave", null);
            },
          },
          e
        );
      },
      getListProps: function (e) {
        var n = e || {},
          r = n.source,
          o = st(n, it);
        return lt(
          {
            role: "listbox",
            "aria-labelledby": ie(t.id, "label"),
            id: ie(t.id, "list", r),
          },
          o
        );
      },
      getItemProps: function (e) {
        var i = e.item,
          u = e.source,
          a = st(e, ut);
        return lt(
          {
            id: ie(t.id, "item-".concat(i.__autocomplete_id), u),
            role: "option",
            "aria-selected": r.getState().activeItemId === i.__autocomplete_id,
            onMouseMove: function (e) {
              if (i.__autocomplete_id !== r.getState().activeItemId) {
                r.dispatch("mousemove", i.__autocomplete_id);
                var t = oe(r.getState());
                if (null !== r.getState().activeItemId && t) {
                  var u = t.item,
                    a = t.itemInputValue,
                    l = t.itemUrl,
                    c = t.source;
                  c.onActive(
                    lt(
                      {
                        event: e,
                        item: u,
                        itemInputValue: a,
                        itemUrl: l,
                        refresh: n,
                        source: c,
                        state: r.getState(),
                      },
                      o
                    )
                  );
                }
              }
            },
            onMouseDown: function (e) {
              e.preventDefault();
            },
            onClick: function (e) {
              var a = u.getItemInputValue({ item: i, state: r.getState() }),
                l = u.getItemUrl({ item: i, state: r.getState() });
              (l
                ? Promise.resolve()
                : $e(
                    lt(
                      {
                        event: e,
                        nextState: { isOpen: !1 },
                        props: t,
                        query: a,
                        refresh: n,
                        store: r,
                      },
                      o
                    )
                  )
              ).then(function () {
                u.onSelect(
                  lt(
                    {
                      event: e,
                      item: i,
                      itemInputValue: a,
                      itemUrl: l,
                      refresh: n,
                      source: u,
                      state: r.getState(),
                    },
                    o
                  )
                );
              });
            },
          },
          a
        );
      },
    };
  }
  function pt(e) {
    return (
      (pt =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      pt(e)
    );
  }
  function mt(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function vt(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? mt(Object(n), !0).forEach(function (t) {
            dt(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : mt(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function dt(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== pt(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== pt(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === pt(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function yt(e) {
    var t,
      n,
      r,
      o,
      i = e.plugins,
      u = e.options,
      a =
        null ===
          (t = ((null === (n = u.__autocomplete_metadata) || void 0 === n
            ? void 0
            : n.userAgents) || [])[0]) || void 0 === t
          ? void 0
          : t.segment,
      l = a
        ? dt(
            {},
            a,
            Object.keys(
              (null === (r = u.__autocomplete_metadata) || void 0 === r
                ? void 0
                : r.options) || {}
            )
          )
        : {};
    return {
      plugins: i.map(function (e) {
        return {
          name: e.name,
          options: Object.keys(e.__autocomplete_pluginOptions || []),
        };
      }),
      options: vt({ "autocomplete-core": Object.keys(u) }, l),
      ua: S.concat(
        (null === (o = u.__autocomplete_metadata) || void 0 === o
          ? void 0
          : o.userAgents) || []
      ),
    };
  }
  function bt(e) {
    var t,
      n = e.state;
    return !1 === n.isOpen || null === n.activeItemId
      ? null
      : (null === (t = oe(n)) || void 0 === t ? void 0 : t.itemInputValue) ||
          null;
  }
  function gt(e) {
    return (
      (gt =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      gt(e)
    );
  }
  function ht(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function Ot(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? ht(Object(n), !0).forEach(function (t) {
            _t(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : ht(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function _t(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== gt(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== gt(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === gt(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  var St = function (e, t) {
    switch (t.type) {
      case "setActiveItemId":
      case "mousemove":
        return Ot(Ot({}, e), {}, { activeItemId: t.payload });
      case "setQuery":
        return Ot(Ot({}, e), {}, { query: t.payload, completion: null });
      case "setCollections":
        return Ot(Ot({}, e), {}, { collections: t.payload });
      case "setIsOpen":
        return Ot(Ot({}, e), {}, { isOpen: t.payload });
      case "setStatus":
        return Ot(Ot({}, e), {}, { status: t.payload });
      case "setContext":
        return Ot(Ot({}, e), {}, { context: Ot(Ot({}, e.context), t.payload) });
      case "ArrowDown":
        var n = Ot(
          Ot({}, e),
          {},
          {
            activeItemId: t.payload.hasOwnProperty("nextActiveItemId")
              ? t.payload.nextActiveItemId
              : Z(1, e.activeItemId, b(e), t.props.defaultActiveItemId),
          }
        );
        return Ot(Ot({}, n), {}, { completion: bt({ state: n }) });
      case "ArrowUp":
        var r = Ot(
          Ot({}, e),
          {},
          {
            activeItemId: Z(
              -1,
              e.activeItemId,
              b(e),
              t.props.defaultActiveItemId
            ),
          }
        );
        return Ot(Ot({}, r), {}, { completion: bt({ state: r }) });
      case "Escape":
        return e.isOpen
          ? Ot(
              Ot({}, e),
              {},
              { activeItemId: null, isOpen: !1, completion: null }
            )
          : Ot(
              Ot({}, e),
              {},
              { activeItemId: null, query: "", status: "idle", collections: [] }
            );
      case "submit":
        return Ot(
          Ot({}, e),
          {},
          { activeItemId: null, isOpen: !1, status: "idle" }
        );
      case "reset":
        return Ot(
          Ot({}, e),
          {},
          {
            activeItemId:
              !0 === t.props.openOnFocus ? t.props.defaultActiveItemId : null,
            status: "idle",
            query: "",
          }
        );
      case "focus":
        return Ot(
          Ot({}, e),
          {},
          {
            activeItemId: t.props.defaultActiveItemId,
            isOpen:
              (t.props.openOnFocus || Boolean(e.query)) &&
              t.props.shouldPanelOpen({ state: e }),
          }
        );
      case "blur":
        return t.props.debug
          ? e
          : Ot(Ot({}, e), {}, { isOpen: !1, activeItemId: null });
      case "mouseleave":
        return Ot(Ot({}, e), {}, { activeItemId: t.props.defaultActiveItemId });
      default:
        return (
          "The reducer action ".concat(
            JSON.stringify(t.type),
            " is not supported."
          ),
          e
        );
    }
  };
  function jt(e) {
    return (
      (jt =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      jt(e)
    );
  }
  function Pt(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function wt(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? Pt(Object(n), !0).forEach(function (t) {
            It(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : Pt(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function It(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== jt(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== jt(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === jt(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function At(e) {
    var t = [],
      n = _e(e, t),
      r = se(St, n, function (e) {
        var t = e.prevState,
          r = e.state;
        n.onStateChange(
          wt({ prevState: t, state: r, refresh: u, navigator: n.navigator }, o)
        );
      }),
      o = (function (e) {
        var t = e.store;
        return {
          setActiveItemId: function (e) {
            t.dispatch("setActiveItemId", e);
          },
          setQuery: function (e) {
            t.dispatch("setQuery", e);
          },
          setCollections: function (e) {
            var n = 0,
              r = e.map(function (e) {
                return me(
                  me({}, e),
                  {},
                  {
                    items: m(e.items).map(function (e) {
                      return me(me({}, e), {}, { __autocomplete_id: n++ });
                    }),
                  }
                );
              });
            t.dispatch("setCollections", r);
          },
          setIsOpen: function (e) {
            t.dispatch("setIsOpen", e);
          },
          setStatus: function (e) {
            t.dispatch("setStatus", e);
          },
          setContext: function (e) {
            t.dispatch("setContext", e);
          },
        };
      })({ store: r }),
      i = ft(wt({ props: n, refresh: u, store: r, navigator: n.navigator }, o));
    function u() {
      return $e(
        wt(
          {
            event: new Event("input"),
            nextState: { isOpen: r.getState().isOpen },
            props: n,
            navigator: n.navigator,
            query: r.getState().query,
            refresh: u,
            store: r,
          },
          o
        )
      );
    }
    if (
      e.insights &&
      !n.plugins.some(function (e) {
        return "aa.algoliaInsightsPlugin" === e.name;
      })
    ) {
      var a = "boolean" == typeof e.insights ? {} : e.insights;
      n.plugins.push(G(a));
    }
    return (
      n.plugins.forEach(function (e) {
        var r;
        return null === (r = e.subscribe) || void 0 === r
          ? void 0
          : r.call(
              e,
              wt(
                wt({}, o),
                {},
                {
                  navigator: n.navigator,
                  refresh: u,
                  onSelect: function (e) {
                    t.push({ onSelect: e });
                  },
                  onActive: function (e) {
                    t.push({ onActive: e });
                  },
                  onResolve: function (e) {
                    t.push({ onResolve: e });
                  },
                }
              )
            );
      }),
      (function (e) {
        var t,
          n,
          r = e.metadata,
          o = e.environment;
        if (
          null === (t = o.navigator) ||
          void 0 === t ||
          null === (n = t.userAgent) ||
          void 0 === n
            ? void 0
            : n.includes("Algolia Crawler")
        ) {
          var i = o.document.createElement("meta"),
            u = o.document.querySelector("head");
          (i.name = "algolia:metadata"),
            setTimeout(function () {
              (i.content = JSON.stringify(r)), u.appendChild(i);
            }, 0);
        }
      })({
        metadata: yt({ plugins: n.plugins, options: e }),
        environment: n.environment,
      }),
      wt(wt({ refresh: u, navigator: n.navigator }, i), o)
    );
  }
  var Et = function (e, t, n, r) {
      var o;
      t[0] = 0;
      for (var i = 1; i < t.length; i++) {
        var u = t[i++],
          a = t[i] ? ((t[0] |= u ? 1 : 2), n[t[i++]]) : t[++i];
        3 === u
          ? (r[0] = a)
          : 4 === u
          ? (r[1] = Object.assign(r[1] || {}, a))
          : 5 === u
          ? ((r[1] = r[1] || {})[t[++i]] = a)
          : 6 === u
          ? (r[1][t[++i]] += a + "")
          : u
          ? ((o = e.apply(a, Et(e, a, n, ["", null]))),
            r.push(o),
            a[0] ? (t[0] |= 2) : ((t[i - 2] = 0), (t[i] = o)))
          : r.push(a);
      }
      return r;
    },
    Dt = new Map();
  function Ct(e) {
    var t = Dt.get(this);
    return (
      t || ((t = new Map()), Dt.set(this, t)),
      (t = Et(
        this,
        t.get(e) ||
          (t.set(
            e,
            (t = (function (e) {
              for (
                var t,
                  n,
                  r = 1,
                  o = "",
                  i = "",
                  u = [0],
                  a = function (e) {
                    1 === r &&
                    (e || (o = o.replace(/^\s*\n\s*|\s*\n\s*$/g, "")))
                      ? u.push(0, e, o)
                      : 3 === r && (e || o)
                      ? (u.push(3, e, o), (r = 2))
                      : 2 === r && "..." === o && e
                      ? u.push(4, e, 0)
                      : 2 === r && o && !e
                      ? u.push(5, 0, !0, o)
                      : r >= 5 &&
                        ((o || (!e && 5 === r)) &&
                          (u.push(r, 0, o, n), (r = 6)),
                        e && (u.push(r, e, 0, n), (r = 6))),
                      (o = "");
                  },
                  l = 0;
                l < e.length;
                l++
              ) {
                l && (1 === r && a(), a(l));
                for (var c = 0; c < e[l].length; c++)
                  (t = e[l][c]),
                    1 === r
                      ? "<" === t
                        ? (a(), (u = [u]), (r = 3))
                        : (o += t)
                      : 4 === r
                      ? "--" === o && ">" === t
                        ? ((r = 1), (o = ""))
                        : (o = t + o[0])
                      : i
                      ? t === i
                        ? (i = "")
                        : (o += t)
                      : '"' === t || "'" === t
                      ? (i = t)
                      : ">" === t
                      ? (a(), (r = 1))
                      : r &&
                        ("=" === t
                          ? ((r = 5), (n = o), (o = ""))
                          : "/" === t && (r < 5 || ">" === e[l][c + 1])
                          ? (a(),
                            3 === r && (u = u[0]),
                            (r = u),
                            (u = u[0]).push(2, 0, r),
                            (r = 0))
                          : " " === t || "\t" === t || "\n" === t || "\r" === t
                          ? (a(), (r = 2))
                          : (o += t)),
                    3 === r && "!--" === o && ((r = 4), (u = u[0]));
              }
              return a(), u;
            })(e))
          ),
          t),
        arguments,
        []
      )).length > 1
        ? t
        : t[0]
    );
  }
  var kt = function (e) {
    var t = e.environment,
      n = t.document.createElementNS("http://www.w3.org/2000/svg", "svg");
    n.setAttribute("class", "aa-ClearIcon"),
      n.setAttribute("viewBox", "0 0 24 24"),
      n.setAttribute("width", "18"),
      n.setAttribute("height", "18"),
      n.setAttribute("fill", "currentColor");
    var r = t.document.createElementNS("http://www.w3.org/2000/svg", "path");
    return (
      r.setAttribute(
        "d",
        "M5.293 6.707l5.293 5.293-5.293 5.293c-0.391 0.391-0.391 1.024 0 1.414s1.024 0.391 1.414 0l5.293-5.293 5.293 5.293c0.391 0.391 1.024 0.391 1.414 0s0.391-1.024 0-1.414l-5.293-5.293 5.293-5.293c0.391-0.391 0.391-1.024 0-1.414s-1.024-0.391-1.414 0l-5.293 5.293-5.293-5.293c-0.391-0.391-1.024-0.391-1.414 0s-0.391 1.024 0 1.414z"
      ),
      n.appendChild(r),
      n
    );
  };
  function xt(e, t) {
    if ("string" == typeof t) {
      var n = e.document.querySelector(t);
      return (
        "The element ".concat(JSON.stringify(t), " is not in the document."), n
      );
    }
    return t;
  }
  function Nt() {
    for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
      t[n] = arguments[n];
    return t.reduce(function (e, t) {
      return (
        Object.keys(t).forEach(function (n) {
          var r = e[n],
            o = t[n];
          r !== o && (e[n] = [r, o].filter(Boolean).join(" "));
        }),
        e
      );
    }, {});
  }
  var Tt = function (e) {
    return (
      e &&
      "object" === r(e) &&
      "[object Object]" === Object.prototype.toString.call(e)
    );
  };
  function qt() {
    for (var e = arguments.length, t = new Array(e), n = 0; n < e; n++)
      t[n] = arguments[n];
    return t.reduce(function (e, t) {
      return (
        Object.keys(t).forEach(function (n) {
          var r = e[n],
            o = t[n];
          Array.isArray(r) && Array.isArray(o)
            ? (e[n] = r.concat.apply(r, l(o)))
            : Tt(r) && Tt(o)
            ? (e[n] = qt(r, o))
            : (e[n] = o);
        }),
        e
      );
    }, {});
  }
  function Bt(e, t) {
    return Object.entries(e).reduce(function (e, r) {
      var i = a(r, 2),
        u = i[0],
        l = i[1];
      return t({ key: u, value: l }) ? n(n({}, e), {}, o({}, u, l)) : e;
    }, {});
  }
  var Rt = ["ontouchstart", "ontouchend", "ontouchmove", "ontouchcancel"];
  function Ft(e, t, n) {
    e[t] = null === n ? "" : "number" != typeof n ? n : n + "px";
  }
  function Lt(e) {
    this._listeners[e.type](e);
  }
  function Ut(e, t, n) {
    var r,
      o,
      i = e[t];
    if ("style" === t)
      if ("string" == typeof n) e.style = n;
      else if (null === n) e.style = "";
      else for (t in n) (i && n[t] === i[t]) || Ft(e.style, t, n[t]);
    else
      "o" === t[0] && "n" === t[1]
        ? ((r = t !== (t = t.replace(/Capture$/, ""))),
          ((o = t.toLowerCase()) in e || Rt.includes(o)) && (t = o),
          (t = t.slice(2)),
          e._listeners || (e._listeners = {}),
          (e._listeners[t] = n),
          n
            ? i || e.addEventListener(t, Lt, r)
            : e.removeEventListener(t, Lt, r))
        : "list" !== t &&
          "tagName" !== t &&
          "form" !== t &&
          "type" !== t &&
          "size" !== t &&
          "download" !== t &&
          "href" !== t &&
          t in e
        ? (e[t] = null == n ? "" : n)
        : "function" != typeof n &&
          "dangerouslySetInnerHTML" !== t &&
          (null == n || (!1 === n && !/^ar/.test(t))
            ? e.removeAttribute(t)
            : e.setAttribute(t, n));
  }
  function Mt(e) {
    return "onChange" === e ? "onInput" : e;
  }
  function Ht(e, t) {
    for (var n in t) Ut(e, Mt(n), t[n]);
  }
  function Vt(e, t) {
    for (var n in t) ("o" === n[0] && "n" === n[1]) || Ut(e, Mt(n), t[n]);
  }
  var Wt = ["children"];
  function Kt(e) {
    return function (t, n) {
      var r = n.children,
        o = void 0 === r ? [] : r,
        i = u(n, Wt),
        a = e.document.createElement(t);
      return Ht(a, i), a.append.apply(a, l(o)), a;
    };
  }
  var Qt = [
      "autocompleteScopeApi",
      "environment",
      "classNames",
      "getInputProps",
      "getInputPropsCore",
      "isDetached",
      "state",
    ],
    $t = function (e) {
      var t = e.environment.document.createElementNS(
        "http://www.w3.org/2000/svg",
        "svg"
      );
      return (
        t.setAttribute("class", "aa-LoadingIcon"),
        t.setAttribute("viewBox", "0 0 100 100"),
        t.setAttribute("width", "20"),
        t.setAttribute("height", "20"),
        (t.innerHTML =
          '<circle\n  cx="50"\n  cy="50"\n  fill="none"\n  r="35"\n  stroke="currentColor"\n  stroke-dasharray="164.93361431346415 56.97787143782138"\n  stroke-width="6"\n>\n  <animateTransform\n    attributeName="transform"\n    type="rotate"\n    repeatCount="indefinite"\n    dur="1s"\n    values="0 50 50;90 50 50;180 50 50;360 50 50"\n    keyTimes="0;0.40;0.65;1"\n  />\n</circle>'),
        t
      );
    },
    zt = function (e) {
      var t = e.environment,
        n = t.document.createElementNS("http://www.w3.org/2000/svg", "svg");
      n.setAttribute("class", "aa-SubmitIcon"),
        n.setAttribute("viewBox", "0 0 24 24"),
        n.setAttribute("width", "20"),
        n.setAttribute("height", "20"),
        n.setAttribute("fill", "currentColor");
      var r = t.document.createElementNS("http://www.w3.org/2000/svg", "path");
      return (
        r.setAttribute(
          "d",
          "M16.041 15.856c-0.034 0.026-0.067 0.055-0.099 0.087s-0.060 0.064-0.087 0.099c-1.258 1.213-2.969 1.958-4.855 1.958-1.933 0-3.682-0.782-4.95-2.050s-2.050-3.017-2.050-4.95 0.782-3.682 2.050-4.95 3.017-2.050 4.95-2.050 3.682 0.782 4.95 2.050 2.050 3.017 2.050 4.95c0 1.886-0.745 3.597-1.959 4.856zM21.707 20.293l-3.675-3.675c1.231-1.54 1.968-3.493 1.968-5.618 0-2.485-1.008-4.736-2.636-6.364s-3.879-2.636-6.364-2.636-4.736 1.008-6.364 2.636-2.636 3.879-2.636 6.364 1.008 4.736 2.636 6.364 3.879 2.636 6.364 2.636c2.125 0 4.078-0.737 5.618-1.968l3.675 3.675c0.391 0.391 1.024 0.391 1.414 0s0.391-1.024 0-1.414z"
        ),
        n.appendChild(r),
        n
      );
    };
  function Gt(e) {
    var t = e.autocomplete,
      r = e.autocompleteScopeApi,
      o = e.classNames,
      i = e.environment,
      a = e.isDetached,
      l = e.placeholder,
      c = void 0 === l ? "Search" : l,
      s = e.propGetters,
      f = e.setIsModalOpen,
      p = e.state,
      m = e.translations,
      v = Kt(i),
      d = s.getRootProps(n({ state: p, props: t.getRootProps({}) }, r)),
      y = v("div", n({ class: o.root }, d)),
      b = v("div", {
        class: o.detachedContainer,
        onMouseDown: function (e) {
          e.stopPropagation();
        },
      }),
      g = v("div", {
        class: o.detachedOverlay,
        children: [b],
        onMouseDown: function () {
          f(!1), t.setIsOpen(!1);
        },
      }),
      h = s.getLabelProps(n({ state: p, props: t.getLabelProps({}) }, r)),
      O = v("button", {
        class: o.submitButton,
        type: "submit",
        title: m.submitButtonTitle,
        children: [zt({ environment: i })],
      }),
      _ = v("label", n({ class: o.label, children: [O] }, h)),
      S = v("button", {
        class: o.clearButton,
        type: "reset",
        title: m.clearButtonTitle,
        children: [kt({ environment: i })],
      }),
      j = v("div", {
        class: o.loadingIndicator,
        children: [$t({ environment: i })],
      }),
      P = (function (e) {
        var t = e.autocompleteScopeApi,
          r = e.environment;
        e.classNames;
        var o = e.getInputProps,
          i = e.getInputPropsCore,
          a = e.isDetached,
          l = e.state,
          c = u(e, Qt),
          s = Kt(r)("input", c),
          f = o(
            n({ state: l, props: i({ inputElement: s }), inputElement: s }, t)
          );
        return (
          Ht(
            s,
            n(
              n({}, f),
              {},
              {
                onKeyDown: function (e) {
                  (a && "Tab" === e.key) || f.onKeyDown(e);
                },
              }
            )
          ),
          s
        );
      })({
        class: o.input,
        environment: i,
        state: p,
        getInputProps: s.getInputProps,
        getInputPropsCore: t.getInputProps,
        autocompleteScopeApi: r,
        isDetached: a,
      }),
      w = v("div", { class: o.inputWrapperPrefix, children: [_, j] }),
      I = v("div", { class: o.inputWrapperSuffix, children: [S] }),
      A = v("div", { class: o.inputWrapper, children: [P] }),
      E = s.getFormProps(
        n({ state: p, props: t.getFormProps({ inputElement: P }) }, r)
      ),
      D = v("form", n({ class: o.form, children: [w, A, I] }, E)),
      C = s.getPanelProps(n({ state: p, props: t.getPanelProps({}) }, r)),
      k = v("div", n({ class: o.panel }, C)),
      x = v("div", {
        class: o.detachedSearchButtonQuery,
        textContent: p.query,
      }),
      N = v("div", {
        class: o.detachedSearchButtonPlaceholder,
        hidden: Boolean(p.query),
        textContent: c,
      });
    if (a) {
      var T = v("div", {
          class: o.detachedSearchButtonIcon,
          children: [zt({ environment: i })],
        }),
        q = v("button", {
          type: "button",
          class: o.detachedSearchButton,
          onClick: function () {
            f(!0);
          },
          children: [T, N, x],
        }),
        B = v("button", {
          type: "button",
          class: o.detachedCancelButton,
          textContent: m.detachedCancelButtonText,
          onTouchStart: function (e) {
            e.stopPropagation();
          },
          onClick: function () {
            t.setIsOpen(!1), f(!1);
          },
        }),
        R = v("div", { class: o.detachedFormContainer, children: [D, B] });
      b.appendChild(R), y.appendChild(q);
    } else y.appendChild(D);
    return {
      detachedContainer: b,
      detachedOverlay: g,
      detachedSearchButtonQuery: x,
      detachedSearchButtonPlaceholder: N,
      inputWrapper: A,
      input: P,
      root: y,
      form: D,
      label: _,
      submitButton: O,
      clearButton: S,
      loadingIndicator: j,
      panel: k,
    };
  }
  var Jt,
    Xt,
    Yt,
    Zt,
    en,
    tn,
    nn,
    rn = {},
    on = [],
    un = /acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i;
  function an(e, t) {
    for (var n in t) e[n] = t[n];
    return e;
  }
  function ln(e) {
    var t = e.parentNode;
    t && t.removeChild(e);
  }
  function cn(e, t, n) {
    var r,
      o,
      i,
      u = {};
    for (i in t)
      "key" == i ? (r = t[i]) : "ref" == i ? (o = t[i]) : (u[i] = t[i]);
    if (
      (arguments.length > 2 &&
        (u.children = arguments.length > 3 ? Jt.call(arguments, 2) : n),
      "function" == typeof e && null != e.defaultProps)
    )
      for (i in e.defaultProps) void 0 === u[i] && (u[i] = e.defaultProps[i]);
    return sn(e, u, r, o, null);
  }
  function sn(e, t, n, r, o) {
    var i = {
      type: e,
      props: t,
      key: n,
      ref: r,
      __k: null,
      __: null,
      __b: 0,
      __e: null,
      __d: void 0,
      __c: null,
      __h: null,
      constructor: void 0,
      __v: null == o ? ++Yt : o,
    };
    return null == o && null != Xt.vnode && Xt.vnode(i), i;
  }
  function fn(e) {
    return e.children;
  }
  function pn(e, t) {
    (this.props = e), (this.context = t);
  }
  function mn(e, t) {
    if (null == t) return e.__ ? mn(e.__, e.__.__k.indexOf(e) + 1) : null;
    for (var n; t < e.__k.length; t++)
      if (null != (n = e.__k[t]) && null != n.__e) return n.__e;
    return "function" == typeof e.type ? mn(e) : null;
  }
  function vn(e) {
    var t, n;
    if (null != (e = e.__) && null != e.__c) {
      for (e.__e = e.__c.base = null, t = 0; t < e.__k.length; t++)
        if (null != (n = e.__k[t]) && null != n.__e) {
          e.__e = e.__c.base = n.__e;
          break;
        }
      return vn(e);
    }
  }
  function dn(e) {
    ((!e.__d && (e.__d = !0) && Zt.push(e) && !yn.__r++) ||
      en !== Xt.debounceRendering) &&
      ((en = Xt.debounceRendering) || tn)(yn);
  }
  function yn() {
    var e, t, n, r, o, i, u, a;
    for (Zt.sort(nn); (e = Zt.shift()); )
      e.__d &&
        ((t = Zt.length),
        (r = void 0),
        (o = void 0),
        (u = (i = (n = e).__v).__e),
        (a = n.__P) &&
          ((r = []),
          ((o = an({}, i)).__v = i.__v + 1),
          wn(
            a,
            i,
            o,
            n.__n,
            void 0 !== a.ownerSVGElement,
            null != i.__h ? [u] : null,
            r,
            null == u ? mn(i) : u,
            i.__h
          ),
          In(r, i),
          i.__e != u && vn(i)),
        Zt.length > t && Zt.sort(nn));
    yn.__r = 0;
  }
  function bn(e, t, n, r, o, i, u, a, l, c) {
    var s,
      f,
      p,
      m,
      v,
      d,
      y,
      b = (r && r.__k) || on,
      g = b.length;
    for (n.__k = [], s = 0; s < t.length; s++)
      if (
        null !=
        (m = n.__k[s] =
          null == (m = t[s]) || "boolean" == typeof m || "function" == typeof m
            ? null
            : "string" == typeof m ||
              "number" == typeof m ||
              "bigint" == typeof m
            ? sn(null, m, null, null, m)
            : Array.isArray(m)
            ? sn(fn, { children: m }, null, null, null)
            : m.__b > 0
            ? sn(m.type, m.props, m.key, m.ref ? m.ref : null, m.__v)
            : m)
      ) {
        if (
          ((m.__ = n),
          (m.__b = n.__b + 1),
          null === (p = b[s]) || (p && m.key == p.key && m.type === p.type))
        )
          b[s] = void 0;
        else
          for (f = 0; f < g; f++) {
            if ((p = b[f]) && m.key == p.key && m.type === p.type) {
              b[f] = void 0;
              break;
            }
            p = null;
          }
        wn(e, m, (p = p || rn), o, i, u, a, l, c),
          (v = m.__e),
          (f = m.ref) &&
            p.ref != f &&
            (y || (y = []),
            p.ref && y.push(p.ref, null, m),
            y.push(f, m.__c || v, m)),
          null != v
            ? (null == d && (d = v),
              "function" == typeof m.type && m.__k === p.__k
                ? (m.__d = l = gn(m, l, e))
                : (l = hn(e, m, p, b, v, l)),
              "function" == typeof n.type && (n.__d = l))
            : l && p.__e == l && l.parentNode != e && (l = mn(p));
      }
    for (n.__e = d, s = g; s--; )
      null != b[s] &&
        ("function" == typeof n.type &&
          null != b[s].__e &&
          b[s].__e == n.__d &&
          (n.__d = On(r).nextSibling),
        Dn(b[s], b[s]));
    if (y) for (s = 0; s < y.length; s++) En(y[s], y[++s], y[++s]);
  }
  function gn(e, t, n) {
    for (var r, o = e.__k, i = 0; o && i < o.length; i++)
      (r = o[i]) &&
        ((r.__ = e),
        (t =
          "function" == typeof r.type
            ? gn(r, t, n)
            : hn(n, r, r, o, r.__e, t)));
    return t;
  }
  function hn(e, t, n, r, o, i) {
    var u, a, l;
    if (void 0 !== t.__d) (u = t.__d), (t.__d = void 0);
    else if (null == n || o != i || null == o.parentNode)
      e: if (null == i || i.parentNode !== e) e.appendChild(o), (u = null);
      else {
        for (a = i, l = 0; (a = a.nextSibling) && l < r.length; l += 1)
          if (a == o) break e;
        e.insertBefore(o, i), (u = i);
      }
    return void 0 !== u ? u : o.nextSibling;
  }
  function On(e) {
    var t, n, r;
    if (null == e.type || "string" == typeof e.type) return e.__e;
    if (e.__k)
      for (t = e.__k.length - 1; t >= 0; t--)
        if ((n = e.__k[t]) && (r = On(n))) return r;
    return null;
  }
  function _n(e, t, n) {
    "-" === t[0]
      ? e.setProperty(t, null == n ? "" : n)
      : (e[t] =
          null == n ? "" : "number" != typeof n || un.test(t) ? n : n + "px");
  }
  function Sn(e, t, n, r, o) {
    var i;
    e: if ("style" === t)
      if ("string" == typeof n) e.style.cssText = n;
      else {
        if (("string" == typeof r && (e.style.cssText = r = ""), r))
          for (t in r) (n && t in n) || _n(e.style, t, "");
        if (n) for (t in n) (r && n[t] === r[t]) || _n(e.style, t, n[t]);
      }
    else if ("o" === t[0] && "n" === t[1])
      (i = t !== (t = t.replace(/Capture$/, ""))),
        (t = t.toLowerCase() in e ? t.toLowerCase().slice(2) : t.slice(2)),
        e.l || (e.l = {}),
        (e.l[t + i] = n),
        n
          ? r || e.addEventListener(t, i ? Pn : jn, i)
          : e.removeEventListener(t, i ? Pn : jn, i);
    else if ("dangerouslySetInnerHTML" !== t) {
      if (o) t = t.replace(/xlink(H|:h)/, "h").replace(/sName$/, "s");
      else if (
        "width" !== t &&
        "height" !== t &&
        "href" !== t &&
        "list" !== t &&
        "form" !== t &&
        "tabIndex" !== t &&
        "download" !== t &&
        t in e
      )
        try {
          e[t] = null == n ? "" : n;
          break e;
        } catch (e) {}
      "function" == typeof n ||
        (null == n || (!1 === n && "-" !== t[4])
          ? e.removeAttribute(t)
          : e.setAttribute(t, n));
    }
  }
  function jn(e) {
    return this.l[e.type + !1](Xt.event ? Xt.event(e) : e);
  }
  function Pn(e) {
    return this.l[e.type + !0](Xt.event ? Xt.event(e) : e);
  }
  function wn(e, t, n, r, o, i, u, a, l) {
    var c,
      s,
      f,
      p,
      m,
      v,
      d,
      y,
      b,
      g,
      h,
      O,
      _,
      S,
      j,
      P = t.type;
    if (void 0 !== t.constructor) return null;
    null != n.__h &&
      ((l = n.__h), (a = t.__e = n.__e), (t.__h = null), (i = [a])),
      (c = Xt.__b) && c(t);
    try {
      e: if ("function" == typeof P) {
        if (
          ((y = t.props),
          (b = (c = P.contextType) && r[c.__c]),
          (g = c ? (b ? b.props.value : c.__) : r),
          n.__c
            ? (d = (s = t.__c = n.__c).__ = s.__E)
            : ("prototype" in P && P.prototype.render
                ? (t.__c = s = new P(y, g))
                : ((t.__c = s = new pn(y, g)),
                  (s.constructor = P),
                  (s.render = Cn)),
              b && b.sub(s),
              (s.props = y),
              s.state || (s.state = {}),
              (s.context = g),
              (s.__n = r),
              (f = s.__d = !0),
              (s.__h = []),
              (s._sb = [])),
          null == s.__s && (s.__s = s.state),
          null != P.getDerivedStateFromProps &&
            (s.__s == s.state && (s.__s = an({}, s.__s)),
            an(s.__s, P.getDerivedStateFromProps(y, s.__s))),
          (p = s.props),
          (m = s.state),
          (s.__v = t),
          f)
        )
          null == P.getDerivedStateFromProps &&
            null != s.componentWillMount &&
            s.componentWillMount(),
            null != s.componentDidMount && s.__h.push(s.componentDidMount);
        else {
          if (
            (null == P.getDerivedStateFromProps &&
              y !== p &&
              null != s.componentWillReceiveProps &&
              s.componentWillReceiveProps(y, g),
            (!s.__e &&
              null != s.shouldComponentUpdate &&
              !1 === s.shouldComponentUpdate(y, s.__s, g)) ||
              t.__v === n.__v)
          ) {
            for (
              t.__v !== n.__v &&
                ((s.props = y), (s.state = s.__s), (s.__d = !1)),
                s.__e = !1,
                t.__e = n.__e,
                t.__k = n.__k,
                t.__k.forEach(function (e) {
                  e && (e.__ = t);
                }),
                h = 0;
              h < s._sb.length;
              h++
            )
              s.__h.push(s._sb[h]);
            (s._sb = []), s.__h.length && u.push(s);
            break e;
          }
          null != s.componentWillUpdate && s.componentWillUpdate(y, s.__s, g),
            null != s.componentDidUpdate &&
              s.__h.push(function () {
                s.componentDidUpdate(p, m, v);
              });
        }
        if (
          ((s.context = g),
          (s.props = y),
          (s.__P = e),
          (O = Xt.__r),
          (_ = 0),
          "prototype" in P && P.prototype.render)
        ) {
          for (
            s.state = s.__s,
              s.__d = !1,
              O && O(t),
              c = s.render(s.props, s.state, s.context),
              S = 0;
            S < s._sb.length;
            S++
          )
            s.__h.push(s._sb[S]);
          s._sb = [];
        } else
          do {
            (s.__d = !1),
              O && O(t),
              (c = s.render(s.props, s.state, s.context)),
              (s.state = s.__s);
          } while (s.__d && ++_ < 25);
        (s.state = s.__s),
          null != s.getChildContext && (r = an(an({}, r), s.getChildContext())),
          f ||
            null == s.getSnapshotBeforeUpdate ||
            (v = s.getSnapshotBeforeUpdate(p, m)),
          (j =
            null != c && c.type === fn && null == c.key ? c.props.children : c),
          bn(e, Array.isArray(j) ? j : [j], t, n, r, o, i, u, a, l),
          (s.base = t.__e),
          (t.__h = null),
          s.__h.length && u.push(s),
          d && (s.__E = s.__ = null),
          (s.__e = !1);
      } else null == i && t.__v === n.__v ? ((t.__k = n.__k), (t.__e = n.__e)) : (t.__e = An(n.__e, t, n, r, o, i, u, l));
      (c = Xt.diffed) && c(t);
    } catch (e) {
      (t.__v = null),
        (l || null != i) &&
          ((t.__e = a), (t.__h = !!l), (i[i.indexOf(a)] = null)),
        Xt.__e(e, t, n);
    }
  }
  function In(e, t) {
    Xt.__c && Xt.__c(t, e),
      e.some(function (t) {
        try {
          (e = t.__h),
            (t.__h = []),
            e.some(function (e) {
              e.call(t);
            });
        } catch (e) {
          Xt.__e(e, t.__v);
        }
      });
  }
  function An(e, t, n, r, o, i, u, a) {
    var l,
      c,
      s,
      f = n.props,
      p = t.props,
      m = t.type,
      v = 0;
    if (("svg" === m && (o = !0), null != i))
      for (; v < i.length; v++)
        if (
          (l = i[v]) &&
          "setAttribute" in l == !!m &&
          (m ? l.localName === m : 3 === l.nodeType)
        ) {
          (e = l), (i[v] = null);
          break;
        }
    if (null == e) {
      if (null === m) return document.createTextNode(p);
      (e = o
        ? document.createElementNS("http://www.w3.org/2000/svg", m)
        : document.createElement(m, p.is && p)),
        (i = null),
        (a = !1);
    }
    if (null === m) f === p || (a && e.data === p) || (e.data = p);
    else {
      if (
        ((i = i && Jt.call(e.childNodes)),
        (c = (f = n.props || rn).dangerouslySetInnerHTML),
        (s = p.dangerouslySetInnerHTML),
        !a)
      ) {
        if (null != i)
          for (f = {}, v = 0; v < e.attributes.length; v++)
            f[e.attributes[v].name] = e.attributes[v].value;
        (s || c) &&
          ((s && ((c && s.__html == c.__html) || s.__html === e.innerHTML)) ||
            (e.innerHTML = (s && s.__html) || ""));
      }
      if (
        ((function (e, t, n, r, o) {
          var i;
          for (i in n)
            "children" === i ||
              "key" === i ||
              i in t ||
              Sn(e, i, null, n[i], r);
          for (i in t)
            (o && "function" != typeof t[i]) ||
              "children" === i ||
              "key" === i ||
              "value" === i ||
              "checked" === i ||
              n[i] === t[i] ||
              Sn(e, i, t[i], n[i], r);
        })(e, p, f, o, a),
        s)
      )
        t.__k = [];
      else if (
        ((v = t.props.children),
        bn(
          e,
          Array.isArray(v) ? v : [v],
          t,
          n,
          r,
          o && "foreignObject" !== m,
          i,
          u,
          i ? i[0] : n.__k && mn(n, 0),
          a
        ),
        null != i)
      )
        for (v = i.length; v--; ) null != i[v] && ln(i[v]);
      a ||
        ("value" in p &&
          void 0 !== (v = p.value) &&
          (v !== e.value ||
            ("progress" === m && !v) ||
            ("option" === m && v !== f.value)) &&
          Sn(e, "value", v, f.value, !1),
        "checked" in p &&
          void 0 !== (v = p.checked) &&
          v !== e.checked &&
          Sn(e, "checked", v, f.checked, !1));
    }
    return e;
  }
  function En(e, t, n) {
    try {
      "function" == typeof e ? e(t) : (e.current = t);
    } catch (e) {
      Xt.__e(e, n);
    }
  }
  function Dn(e, t, n) {
    var r, o;
    if (
      (Xt.unmount && Xt.unmount(e),
      (r = e.ref) && ((r.current && r.current !== e.__e) || En(r, null, t)),
      null != (r = e.__c))
    ) {
      if (r.componentWillUnmount)
        try {
          r.componentWillUnmount();
        } catch (e) {
          Xt.__e(e, t);
        }
      (r.base = r.__P = null), (e.__c = void 0);
    }
    if ((r = e.__k))
      for (o = 0; o < r.length; o++)
        r[o] && Dn(r[o], t, n || "function" != typeof e.type);
    n || null == e.__e || ln(e.__e), (e.__ = e.__e = e.__d = void 0);
  }
  function Cn(e, t, n) {
    return this.constructor(e, n);
  }
  (Jt = on.slice),
    (Xt = {
      __e: function (e, t, n, r) {
        for (var o, i, u; (t = t.__); )
          if ((o = t.__c) && !o.__)
            try {
              if (
                ((i = o.constructor) &&
                  null != i.getDerivedStateFromError &&
                  (o.setState(i.getDerivedStateFromError(e)), (u = o.__d)),
                null != o.componentDidCatch &&
                  (o.componentDidCatch(e, r || {}), (u = o.__d)),
                u)
              )
                return (o.__E = o);
            } catch (t) {
              e = t;
            }
        throw e;
      },
    }),
    (Yt = 0),
    (pn.prototype.setState = function (e, t) {
      var n;
      (n =
        null != this.__s && this.__s !== this.state
          ? this.__s
          : (this.__s = an({}, this.state))),
        "function" == typeof e && (e = e(an({}, n), this.props)),
        e && an(n, e),
        null != e && this.__v && (t && this._sb.push(t), dn(this));
    }),
    (pn.prototype.forceUpdate = function (e) {
      this.__v && ((this.__e = !0), e && this.__h.push(e), dn(this));
    }),
    (pn.prototype.render = fn),
    (Zt = []),
    (tn =
      "function" == typeof Promise
        ? Promise.prototype.then.bind(Promise.resolve())
        : setTimeout),
    (nn = function (e, t) {
      return e.__v.__b - t.__v.__b;
    }),
    (yn.__r = 0);
  var kn = "__aa-highlight__",
    xn = "__/aa-highlight__";
  function Nn(e) {
    var t = e.highlightedValue.split(kn),
      n = t.shift(),
      r = (function () {
        var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : [];
        return {
          get: function () {
            return e;
          },
          add: function (t) {
            var n = e[e.length - 1];
            (null == n ? void 0 : n.isHighlighted) === t.isHighlighted
              ? (e[e.length - 1] = {
                  value: n.value + t.value,
                  isHighlighted: n.isHighlighted,
                })
              : e.push(t);
          },
        };
      })(n ? [{ value: n, isHighlighted: !1 }] : []);
    return (
      t.forEach(function (e) {
        var t = e.split(xn);
        r.add({ value: t[0], isHighlighted: !0 }),
          "" !== t[1] && r.add({ value: t[1], isHighlighted: !1 });
      }),
      r.get()
    );
  }
  function Tn(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return qn(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return qn(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return qn(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function qn(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function Bn(e) {
    var t = e.hit,
      n = e.attribute,
      r = Array.isArray(n) ? n : [n],
      o = y(t, ["_highlightResult"].concat(Tn(r), ["value"]));
    return (
      "string" != typeof o && (o = y(t, r) || ""), Nn({ highlightedValue: o })
    );
  }
  var Rn = {
      "&amp;": "&",
      "&lt;": "<",
      "&gt;": ">",
      "&quot;": '"',
      "&#39;": "'",
    },
    Fn = new RegExp(/\w/i),
    Ln = /&(amp|quot|lt|gt|#39);/g,
    Un = RegExp(Ln.source);
  function Mn(e, t) {
    var n,
      r,
      o,
      i = e[t],
      u =
        (null === (n = e[t + 1]) || void 0 === n ? void 0 : n.isHighlighted) ||
        !0,
      a =
        (null === (r = e[t - 1]) || void 0 === r ? void 0 : r.isHighlighted) ||
        !0;
    return Fn.test(
      (o = i.value) && Un.test(o)
        ? o.replace(Ln, function (e) {
            return Rn[e];
          })
        : o
    ) || a !== u
      ? i.isHighlighted
      : a;
  }
  function Hn(e) {
    return (
      (Hn =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      Hn(e)
    );
  }
  function Vn(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function Wn(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? Vn(Object(n), !0).forEach(function (t) {
            Kn(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : Vn(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function Kn(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== Hn(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== Hn(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === Hn(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function Qn(e) {
    return e.some(function (e) {
      return e.isHighlighted;
    })
      ? e.map(function (t, n) {
          return Wn(Wn({}, t), {}, { isHighlighted: !Mn(e, n) });
        })
      : e.map(function (e) {
          return Wn(Wn({}, e), {}, { isHighlighted: !1 });
        });
  }
  function $n(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return zn(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return zn(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return zn(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function zn(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function Gn(e) {
    var t = e.hit,
      n = e.attribute,
      r = Array.isArray(n) ? n : [n],
      o = y(t, ["_snippetResult"].concat($n(r), ["value"]));
    return (
      "string" != typeof o && (o = y(t, r) || ""), Nn({ highlightedValue: o })
    );
  }
  function Jn(e) {
    return (
      (Jn =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      Jn(e)
    );
  }
  function Xn(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function Yn(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? Xn(Object(n), !0).forEach(function (t) {
            Zn(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : Xn(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function Zn(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== Jn(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== Jn(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === Jn(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function er(e) {
    return (
      (er =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (e) {
              return typeof e;
            }
          : function (e) {
              return e &&
                "function" == typeof Symbol &&
                e.constructor === Symbol &&
                e !== Symbol.prototype
                ? "symbol"
                : typeof e;
            }),
      er(e)
    );
  }
  var tr = ["params"];
  function nr(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var r = Object.getOwnPropertySymbols(e);
      t &&
        (r = r.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, r);
    }
    return n;
  }
  function rr(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? nr(Object(n), !0).forEach(function (t) {
            or(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : nr(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function or(e, t, n) {
    return (
      (t = (function (e) {
        var t = (function (e, t) {
          if ("object" !== er(e) || null === e) return e;
          var n = e[Symbol.toPrimitive];
          if (void 0 !== n) {
            var r = n.call(e, t || "default");
            if ("object" !== er(r)) return r;
            throw new TypeError("@@toPrimitive must return a primitive value.");
          }
          return ("string" === t ? String : Number)(e);
        })(e, "string");
        return "symbol" === er(t) ? t : String(t);
      })(t)) in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function ir(e, t) {
    if (null == e) return {};
    var n,
      r,
      o = (function (e, t) {
        if (null == e) return {};
        var n,
          r,
          o = {},
          i = Object.keys(e);
        for (r = 0; r < i.length; r++)
          (n = i[r]), t.indexOf(n) >= 0 || (o[n] = e[n]);
        return o;
      })(e, t);
    if (Object.getOwnPropertySymbols) {
      var i = Object.getOwnPropertySymbols(e);
      for (r = 0; r < i.length; r++)
        (n = i[r]),
          t.indexOf(n) >= 0 ||
            (Object.prototype.propertyIsEnumerable.call(e, n) && (o[n] = e[n]));
    }
    return o;
  }
  function ur(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return ar(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return ar(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return ar(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."
        );
      })()
    );
  }
  function ar(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, r = new Array(t); n < t; n++) r[n] = e[n];
    return r;
  }
  function lr(e) {
    var t = e.createElement,
      n = e.Fragment;
    function r(e) {
      var r = e.hit,
        o = e.attribute,
        i = e.tagName,
        u = void 0 === i ? "mark" : i;
      return t(
        n,
        {},
        Bn({ hit: r, attribute: o }).map(function (e, n) {
          return e.isHighlighted ? t(u, { key: n }, e.value) : e.value;
        })
      );
    }
    return (r.__autocomplete_componentName = "Highlight"), r;
  }
  function cr(e) {
    var t = e.createElement,
      n = e.Fragment;
    function r(e) {
      var r,
        o = e.hit,
        i = e.attribute,
        u = e.tagName,
        a = void 0 === u ? "mark" : u;
      return t(
        n,
        {},
        ((r = { hit: o, attribute: i }), Qn(Bn(r))).map(function (e, n) {
          return e.isHighlighted ? t(a, { key: n }, e.value) : e.value;
        })
      );
    }
    return (r.__autocomplete_componentName = "ReverseHighlight"), r;
  }
  function sr(e) {
    var t = e.createElement,
      n = e.Fragment;
    function r(e) {
      var r,
        o = e.hit,
        i = e.attribute,
        u = e.tagName,
        a = void 0 === u ? "mark" : u;
      return t(
        n,
        {},
        ((r = { hit: o, attribute: i }), Qn(Gn(r))).map(function (e, n) {
          return e.isHighlighted ? t(a, { key: n }, e.value) : e.value;
        })
      );
    }
    return (r.__autocomplete_componentName = "ReverseSnippet"), r;
  }
  function fr(e) {
    var t = e.createElement,
      n = e.Fragment;
    function r(e) {
      var r = e.hit,
        o = e.attribute,
        i = e.tagName,
        u = void 0 === i ? "mark" : i;
      return t(
        n,
        {},
        Gn({ hit: r, attribute: o }).map(function (e, n) {
          return e.isHighlighted ? t(u, { key: n }, e.value) : e.value;
        })
      );
    }
    return (r.__autocomplete_componentName = "Snippet"), r;
  }
  var pr = [
      "classNames",
      "container",
      "getEnvironmentProps",
      "getFormProps",
      "getInputProps",
      "getItemProps",
      "getLabelProps",
      "getListProps",
      "getPanelProps",
      "getRootProps",
      "panelContainer",
      "panelPlacement",
      "render",
      "renderNoResults",
      "renderer",
      "detachedMediaQuery",
      "components",
      "translations",
    ],
    mr = {
      clearButton: "aa-ClearButton",
      detachedCancelButton: "aa-DetachedCancelButton",
      detachedContainer: "aa-DetachedContainer",
      detachedFormContainer: "aa-DetachedFormContainer",
      detachedOverlay: "aa-DetachedOverlay",
      detachedSearchButton: "aa-DetachedSearchButton",
      detachedSearchButtonIcon: "aa-DetachedSearchButtonIcon",
      detachedSearchButtonPlaceholder: "aa-DetachedSearchButtonPlaceholder",
      detachedSearchButtonQuery: "aa-DetachedSearchButtonQuery",
      form: "aa-Form",
      input: "aa-Input",
      inputWrapper: "aa-InputWrapper",
      inputWrapperPrefix: "aa-InputWrapperPrefix",
      inputWrapperSuffix: "aa-InputWrapperSuffix",
      item: "aa-Item",
      label: "aa-Label",
      list: "aa-List",
      loadingIndicator: "aa-LoadingIndicator",
      panel: "aa-Panel",
      panelLayout: "aa-PanelLayout aa-Panel--scrollable",
      root: "aa-Autocomplete",
      source: "aa-Source",
      sourceFooter: "aa-SourceFooter",
      sourceHeader: "aa-SourceHeader",
      sourceNoResults: "aa-SourceNoResults",
      submitButton: "aa-SubmitButton",
    },
    vr = function (e, t) {
      var n = e.children;
      (0, e.render)(n, t);
    },
    dr = {
      createElement: cn,
      Fragment: fn,
      render: function (e, t, n) {
        var r, o, i;
        Xt.__ && Xt.__(e, t),
          (o = (r = "function" == typeof n) ? null : (n && n.__k) || t.__k),
          (i = []),
          wn(
            t,
            (e = ((!r && n) || t).__k = cn(fn, null, [e])),
            o || rn,
            rn,
            void 0 !== t.ownerSVGElement,
            !r && n
              ? [n]
              : o
              ? null
              : t.firstChild
              ? Jt.call(t.childNodes)
              : null,
            i,
            !r && n ? n : o ? o.__e : t.firstChild,
            r
          ),
          In(i, e);
      },
    };
  function yr(e) {
    var t = e.panelPlacement,
      n = e.container,
      r = e.form,
      o = e.environment,
      i = n.getBoundingClientRect(),
      u =
        (o.pageYOffset ||
          o.document.documentElement.scrollTop ||
          o.document.body.scrollTop ||
          0) +
        i.top +
        i.height;
    switch (t) {
      case "start":
        return { top: u, left: i.left };
      case "end":
        return {
          top: u,
          right: o.document.documentElement.clientWidth - (i.left + i.width),
        };
      case "full-width":
        return { top: u, left: 0, right: 0, width: "unset", maxWidth: "unset" };
      case "input-wrapper-width":
        var a = r.getBoundingClientRect();
        return {
          top: u,
          left: a.left,
          right: o.document.documentElement.clientWidth - (a.left + a.width),
          width: "unset",
          maxWidth: "unset",
        };
      default:
        throw new Error(
          "[Autocomplete] The `panelPlacement` value ".concat(
            JSON.stringify(t),
            " is not valid."
          )
        );
    }
  }
  var br = [{ segment: "autocomplete-js", version: _ }],
    gr = ["components"];
  var hr = (function (e, t) {
    function n(t) {
      return e({
        searchClient: t.searchClient,
        queries: t.requests.map(function (e) {
          return e.query;
        }),
      }).then(function (e) {
        return e.map(function (e, n) {
          var r = t.requests[n];
          return {
            items: e,
            sourceId: r.sourceId,
            transformResponse: r.transformResponse,
          };
        });
      });
    }
    return function (e) {
      return function (r) {
        return Yn(Yn({ requesterId: t, execute: n }, e), r);
      };
    };
  })(function (e) {
    return (function (e) {
      var t = e.searchClient,
        n = e.queries,
        r = e.userAgents,
        o = void 0 === r ? [] : r;
      "function" == typeof t.addAlgoliaAgent &&
        [].concat(ur(S), ur(o)).forEach(function (e) {
          var n = e.segment,
            r = e.version;
          t.addAlgoliaAgent(n, r);
        });
      var i = (function (e) {
          var t = e.transporter || {},
            n = t.headers,
            r = void 0 === n ? {} : n,
            o = t.queryParameters,
            i = void 0 === o ? {} : o,
            u = "x-algolia-application-id",
            a = "x-algolia-api-key";
          return { appId: r[u] || i[u], apiKey: r[a] || i[a] };
        })(t),
        u = i.appId,
        a = i.apiKey;
      return t
        .search(
          n.map(function (e) {
            var t = e.params;
            return rr(
              rr({}, ir(e, tr)),
              {},
              {
                params: rr(
                  { hitsPerPage: 5, highlightPreTag: kn, highlightPostTag: xn },
                  t
                ),
              }
            );
          })
        )
        .then(function (e) {
          return e.results.map(function (e, t) {
            var r;
            return rr(
              rr({}, e),
              {},
              {
                hits:
                  null === (r = e.hits) || void 0 === r
                    ? void 0
                    : r.map(function (r) {
                        return rr(
                          rr({}, r),
                          {},
                          {
                            __autocomplete_indexName: e.index || n[t].indexName,
                            __autocomplete_queryID: e.queryID,
                            __autocomplete_algoliaCredentials: {
                              appId: u,
                              apiKey: a,
                            },
                          }
                        );
                      }),
              }
            );
          });
        });
    })(n(n({}, e), {}, { userAgents: br }));
  }, "algolia");
  var Or = hr({
    transformResponse: function (e) {
      return e.hits;
    },
  });
  (e.autocomplete = function (e) {
    var t,
      r = (function () {
        var e = [],
          t = [];
        function n(n) {
          e.push(n);
          var r = n();
          t.push(r);
        }
        return {
          runEffect: n,
          cleanupEffects: function () {
            var e = t;
            (t = []),
              e.forEach(function (e) {
                e();
              });
          },
          runEffects: function () {
            var t = e;
            (e = []),
              t.forEach(function (e) {
                n(e);
              });
          },
        };
      })(),
      a = r.runEffect,
      l = r.cleanupEffects,
      c = r.runEffects,
      s =
        ((t = []),
        {
          reactive: function (e) {
            var n = e(),
              r = {
                _fn: e,
                _ref: { current: n },
                get value() {
                  return this._ref.current;
                },
                set value(e) {
                  this._ref.current = e;
                },
              };
            return t.push(r), r;
          },
          runReactives: function () {
            t.forEach(function (e) {
              e._ref.current = e._fn();
            });
          },
        }),
      m = s.reactive,
      v = s.runReactives,
      y = f(!1),
      g = f(e),
      h = f(void 0),
      O = m(function () {
        return (function (e) {
          var t,
            r = e.classNames,
            o = e.container,
            i = e.getEnvironmentProps,
            a = e.getFormProps,
            l = e.getInputProps,
            c = e.getItemProps,
            s = e.getLabelProps,
            f = e.getListProps,
            p = e.getPanelProps,
            m = e.getRootProps,
            v = e.panelContainer,
            y = e.panelPlacement,
            b = e.render,
            g = e.renderNoResults,
            h = e.renderer,
            O = e.detachedMediaQuery,
            _ = e.components,
            S = e.translations,
            j = u(e, pr),
            P = "undefined" != typeof window ? window : {},
            w = xt(P, o);
          w.tagName;
          var I = n(n({}, dr), h),
            A = {
              Highlight: lr(I),
              ReverseHighlight: cr(I),
              ReverseSnippet: sr(I),
              Snippet: fr(I),
            };
          return {
            renderer: {
              classNames: Nt(mr, null != r ? r : {}),
              container: w,
              getEnvironmentProps:
                null != i
                  ? i
                  : function (e) {
                      return e.props;
                    },
              getFormProps:
                null != a
                  ? a
                  : function (e) {
                      return e.props;
                    },
              getInputProps:
                null != l
                  ? l
                  : function (e) {
                      return e.props;
                    },
              getItemProps:
                null != c
                  ? c
                  : function (e) {
                      return e.props;
                    },
              getLabelProps:
                null != s
                  ? s
                  : function (e) {
                      return e.props;
                    },
              getListProps:
                null != f
                  ? f
                  : function (e) {
                      return e.props;
                    },
              getPanelProps:
                null != p
                  ? p
                  : function (e) {
                      return e.props;
                    },
              getRootProps:
                null != m
                  ? m
                  : function (e) {
                      return e.props;
                    },
              panelContainer: v ? xt(P, v) : P.document.body,
              panelPlacement: null != y ? y : "input-wrapper-width",
              render: null != b ? b : vr,
              renderNoResults: g,
              renderer: I,
              detachedMediaQuery:
                null != O
                  ? O
                  : getComputedStyle(
                      P.document.documentElement
                    ).getPropertyValue("--aa-detached-media-query"),
              components: n(n({}, A), _),
              translations: n(
                n(
                  {},
                  {
                    clearButtonTitle: "Clear",
                    detachedCancelButtonText: "Cancel",
                    submitButtonTitle: "Submit",
                  }
                ),
                S
              ),
            },
            core: n(
              n({}, j),
              {},
              {
                id: null !== (t = j.id) && void 0 !== t ? t : d(),
                environment: P,
              }
            ),
          };
        })(g.current);
      }),
      _ = m(function () {
        return O.value.core.environment.matchMedia(
          O.value.renderer.detachedMediaQuery
        ).matches;
      }),
      S = m(function () {
        return At(
          n(
            n({}, O.value.core),
            {},
            {
              onStateChange: function (e) {
                var t, n, r;
                (y.current = e.state.collections.some(function (e) {
                  return e.source.templates.noResults;
                })),
                  null === (t = h.current) || void 0 === t || t.call(h, e),
                  null === (n = (r = O.value.core).onStateChange) ||
                    void 0 === n ||
                    n.call(r, e);
              },
              shouldPanelOpen:
                g.current.shouldPanelOpen ||
                function (e) {
                  var t = e.state;
                  if (_.value) return !0;
                  var n = b(t) > 0;
                  if (!O.value.core.openOnFocus && !t.query) return n;
                  var r = Boolean(
                    y.current || O.value.renderer.renderNoResults
                  );
                  return (!n && r) || n;
                },
              __autocomplete_metadata: { userAgents: br, options: e },
            }
          )
        );
      }),
      j = f(
        n(
          {
            collections: [],
            completion: null,
            context: {},
            isOpen: !1,
            query: "",
            activeItemId: null,
            status: "idle",
          },
          O.value.core.initialState
        )
      ),
      P = {
        getEnvironmentProps: O.value.renderer.getEnvironmentProps,
        getFormProps: O.value.renderer.getFormProps,
        getInputProps: O.value.renderer.getInputProps,
        getItemProps: O.value.renderer.getItemProps,
        getLabelProps: O.value.renderer.getLabelProps,
        getListProps: O.value.renderer.getListProps,
        getPanelProps: O.value.renderer.getPanelProps,
        getRootProps: O.value.renderer.getRootProps,
      },
      w = {
        setActiveItemId: S.value.setActiveItemId,
        setQuery: S.value.setQuery,
        setCollections: S.value.setCollections,
        setIsOpen: S.value.setIsOpen,
        setStatus: S.value.setStatus,
        setContext: S.value.setContext,
        refresh: S.value.refresh,
        navigator: S.value.navigator,
      },
      I = m(function () {
        return Ct.bind(O.value.renderer.renderer.createElement);
      }),
      A = m(function () {
        return Gt({
          autocomplete: S.value,
          autocompleteScopeApi: w,
          classNames: O.value.renderer.classNames,
          environment: O.value.core.environment,
          isDetached: _.value,
          placeholder: O.value.core.placeholder,
          propGetters: P,
          setIsModalOpen: k,
          state: j.current,
          translations: O.value.renderer.translations,
        });
      });
    function E() {
      Ht(A.value.panel, {
        style: _.value
          ? {}
          : yr({
              panelPlacement: O.value.renderer.panelPlacement,
              container: A.value.root,
              form: A.value.form,
              environment: O.value.core.environment,
            }),
      });
    }
    function D(e) {
      j.current = e;
      var t = {
          autocomplete: S.value,
          autocompleteScopeApi: w,
          classNames: O.value.renderer.classNames,
          components: O.value.renderer.components,
          container: O.value.renderer.container,
          html: I.value,
          dom: A.value,
          panelContainer: _.value
            ? A.value.detachedContainer
            : O.value.renderer.panelContainer,
          propGetters: P,
          state: j.current,
          renderer: O.value.renderer.renderer,
        },
        r =
          (!b(e) && !y.current && O.value.renderer.renderNoResults) ||
          O.value.renderer.render;
      !(function (e) {
        var t = e.autocomplete,
          r = e.autocompleteScopeApi,
          o = e.dom,
          i = e.propGetters,
          u = e.state;
        Vt(
          o.root,
          i.getRootProps(n({ state: u, props: t.getRootProps({}) }, r))
        ),
          Vt(
            o.input,
            i.getInputProps(
              n(
                {
                  state: u,
                  props: t.getInputProps({ inputElement: o.input }),
                  inputElement: o.input,
                },
                r
              )
            )
          ),
          Ht(o.label, { hidden: "stalled" === u.status }),
          Ht(o.loadingIndicator, { hidden: "stalled" !== u.status }),
          Ht(o.clearButton, { hidden: !u.query }),
          Ht(o.detachedSearchButtonQuery, { textContent: u.query }),
          Ht(o.detachedSearchButtonPlaceholder, { hidden: Boolean(u.query) });
      })(t),
        (function (e, t) {
          var r = t.autocomplete,
            o = t.autocompleteScopeApi,
            u = t.classNames,
            a = t.html,
            l = t.dom,
            c = t.panelContainer,
            s = t.propGetters,
            f = t.state,
            p = t.components,
            m = t.renderer;
          if (f.isOpen) {
            c.contains(l.panel) ||
              "loading" === f.status ||
              c.appendChild(l.panel),
              l.panel.classList.toggle(
                "aa-Panel--stalled",
                "stalled" === f.status
              );
            var v = f.collections
                .filter(function (e) {
                  var t = e.source,
                    n = e.items;
                  return t.templates.noResults || n.length > 0;
                })
                .map(function (e, t) {
                  var l = e.source,
                    c = e.items;
                  return m.createElement(
                    "section",
                    {
                      key: t,
                      className: u.source,
                      "data-autocomplete-source-id": l.sourceId,
                    },
                    l.templates.header &&
                      m.createElement(
                        "div",
                        { className: u.sourceHeader },
                        l.templates.header({
                          components: p,
                          createElement: m.createElement,
                          Fragment: m.Fragment,
                          items: c,
                          source: l,
                          state: f,
                          html: a,
                        })
                      ),
                    l.templates.noResults && 0 === c.length
                      ? m.createElement(
                          "div",
                          { className: u.sourceNoResults },
                          l.templates.noResults({
                            components: p,
                            createElement: m.createElement,
                            Fragment: m.Fragment,
                            source: l,
                            state: f,
                            html: a,
                          })
                        )
                      : m.createElement(
                          "ul",
                          i(
                            { className: u.list },
                            s.getListProps(
                              n(
                                {
                                  state: f,
                                  props: r.getListProps({ source: l }),
                                },
                                o
                              )
                            )
                          ),
                          c.map(function (e) {
                            var t = r.getItemProps({ item: e, source: l });
                            return m.createElement(
                              "li",
                              i(
                                { key: t.id, className: u.item },
                                s.getItemProps(n({ state: f, props: t }, o))
                              ),
                              l.templates.item({
                                components: p,
                                createElement: m.createElement,
                                Fragment: m.Fragment,
                                item: e,
                                state: f,
                                html: a,
                              })
                            );
                          })
                        ),
                    l.templates.footer &&
                      m.createElement(
                        "div",
                        { className: u.sourceFooter },
                        l.templates.footer({
                          components: p,
                          createElement: m.createElement,
                          Fragment: m.Fragment,
                          items: c,
                          source: l,
                          state: f,
                          html: a,
                        })
                      )
                  );
                }),
              d = m.createElement(
                m.Fragment,
                null,
                m.createElement("div", { className: u.panelLayout }, v),
                m.createElement("div", { className: "aa-GradientBottom" })
              ),
              y = v.reduce(function (e, t) {
                return (e[t.props["data-autocomplete-source-id"]] = t), e;
              }, {});
            e(
              n(
                n({ children: d, state: f, sections: v, elements: y }, m),
                {},
                { components: p, html: a },
                o
              ),
              l.panel
            );
          } else c.contains(l.panel) && c.removeChild(l.panel);
        })(r, t);
    }
    function C() {
      var e =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {};
      l();
      var t = O.value.renderer,
        n = t.components,
        r = u(t, gr);
      (g.current = qt(
        r,
        O.value.core,
        {
          components: Bt(n, function (e) {
            return !e.value.hasOwnProperty("__autocomplete_componentName");
          }),
          initialState: j.current,
        },
        e
      )),
        v(),
        c(),
        S.value.refresh().then(function () {
          D(j.current);
        });
    }
    function k(e) {
      requestAnimationFrame(function () {
        var t = O.value.core.environment.document.body.contains(
          A.value.detachedOverlay
        );
        e !== t &&
          (e
            ? (O.value.core.environment.document.body.appendChild(
                A.value.detachedOverlay
              ),
              O.value.core.environment.document.body.classList.add(
                "aa-Detached"
              ),
              A.value.input.focus())
            : (O.value.core.environment.document.body.removeChild(
                A.value.detachedOverlay
              ),
              O.value.core.environment.document.body.classList.remove(
                "aa-Detached"
              )));
      });
    }
    return (
      a(function () {
        var e = S.value.getEnvironmentProps({
          formElement: A.value.form,
          panelElement: A.value.panel,
          inputElement: A.value.input,
        });
        return (
          Ht(O.value.core.environment, e),
          function () {
            Ht(
              O.value.core.environment,
              Object.keys(e).reduce(function (e, t) {
                return n(n({}, e), {}, o({}, t, void 0));
              }, {})
            );
          }
        );
      }),
      a(function () {
        var e = _.value
            ? O.value.core.environment.document.body
            : O.value.renderer.panelContainer,
          t = _.value ? A.value.detachedOverlay : A.value.panel;
        return (
          _.value && j.current.isOpen && k(!0),
          D(j.current),
          function () {
            e.contains(t) && e.removeChild(t);
          }
        );
      }),
      a(function () {
        var e = O.value.renderer.container;
        return (
          e.appendChild(A.value.root),
          function () {
            e.removeChild(A.value.root);
          }
        );
      }),
      a(function () {
        var e = p(function (e) {
          D(e.state);
        }, 0);
        return (
          (h.current = function (t) {
            var n = t.state,
              r = t.prevState;
            (_.value && r.isOpen !== n.isOpen && k(n.isOpen),
            _.value || !n.isOpen || r.isOpen || E(),
            n.query !== r.query) &&
              O.value.core.environment.document
                .querySelectorAll(".aa-Panel--scrollable")
                .forEach(function (e) {
                  0 !== e.scrollTop && (e.scrollTop = 0);
                });
            e({ state: n });
          }),
          function () {
            h.current = void 0;
          }
        );
      }),
      a(function () {
        var e = p(function () {
          var e = _.value;
          (_.value = O.value.core.environment.matchMedia(
            O.value.renderer.detachedMediaQuery
          ).matches),
            e !== _.value ? C({}) : requestAnimationFrame(E);
        }, 20);
        return (
          O.value.core.environment.addEventListener("resize", e),
          function () {
            O.value.core.environment.removeEventListener("resize", e);
          }
        );
      }),
      a(function () {
        if (!_.value) return function () {};
        function e(e) {
          A.value.detachedContainer.classList.toggle(
            "aa-DetachedContainer--modal",
            e
          );
        }
        function t(t) {
          e(t.matches);
        }
        var n = O.value.core.environment.matchMedia(
          getComputedStyle(
            O.value.core.environment.document.documentElement
          ).getPropertyValue("--aa-detached-modal-media-query")
        );
        e(n.matches);
        var r = Boolean(n.addEventListener);
        return (
          r ? n.addEventListener("change", t) : n.addListener(t),
          function () {
            r ? n.removeEventListener("change", t) : n.removeListener(t);
          }
        );
      }),
      a(function () {
        return requestAnimationFrame(E), function () {};
      }),
      n(
        n({}, w),
        {},
        {
          update: C,
          destroy: function () {
            l();
          },
        }
      )
    );
  }),
    (e.getAlgoliaFacets = function (e) {
      var t = hr({
          transformResponse: function (e) {
            return e.facetHits;
          },
        }),
        r = e.queries.map(function (e) {
          return n(n({}, e), {}, { type: "facet" });
        });
      return t(n(n({}, e), {}, { queries: r }));
    }),
    (e.getAlgoliaResults = Or),
    Object.defineProperty(e, "__esModule", { value: !0 });
});
