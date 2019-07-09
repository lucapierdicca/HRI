﻿(define (domain game2)
(:requirements :strips)
(:predicates (at ?a ?x)
             (empty ?x)
             (adj ?x ?y)
             (orie ?a ?x)
	         (rect ?x ?y))
	         
(:action translate
:parameters (?a ?x ?y)
:precondition (and (adj ?x ?y) (empty ?y))
:effect (and  (not (empty ?y)) (empty ?x) (not (at ?a ?x)) (at ?a ?y)))

(:action rotate
:parameters (?a ?x ?y)
:precondition (and (orie ?a ?x) (rect ?x ?y))
:effect (and  (not (orie ?a ?x)) (orie ?a ?y))))