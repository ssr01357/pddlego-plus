(define (domain house-exploration)
	(:requirements :strips :typing)
	(:types location direction)
	(:predicates 
		(door ?loc1 - location ?loc2 - location ?dir - direction)
		(open ?loc1 - location ?loc2 - location ?dir - direction)
		(at ?loc - location)
		(visited ?loc - location)
	)

	(:action open-door
		:parameters (?loc1 - location ?loc2 - location ?dir - direction)
		:precondition (and (door ?loc1 ?loc2 ?dir))
		:effect (open ?loc1 ?loc2 ?dir)
	)

	(:action move
		:parameters (?from - location ?to - location ?dir - direction)
		:precondition (and (at ?from) (open ?from ?to ?dir))
		:effect (and (at ?to) (not (at ?from)) (visited ?to))
	)
)