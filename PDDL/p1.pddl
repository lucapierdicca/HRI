﻿(define (problem p1)
(:domain game2)
(:objects p00 p01 p10 p11 N S O E red green blue)
(:init 
(adj p00 p10) 
(adj p00 p01)
(adj p10 p00)
(adj p10 p11)
(adj p01 p00)
(adj p01 p11)
(adj p11 p01)
(adj p11 p10)
(rect N E)
(rect N O)
(rect O N)
(rect O S)
(rect S O)
(rect S E)
(rect E N)
(rect E S)
(at red p00)
(orie red N)
(at green p01)
(orie green S)
(at blue p11)
(orie blue E)
(empty p10))
(:goal (and (at red p10) (orie red S) (at green p00) (orie green E) (orie blue S))))