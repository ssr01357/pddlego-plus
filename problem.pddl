(define (problem explore-house)
	(:domain house-exploration)
	(:objects 
		kitchen patio backyard driveway street - location
		south west north east - direction
	)
	(:init 
		(at backyard)
		(door kitchen patio south)
		(door patio kitchen north)
		(open patio backyard south)
		(open kitchen patio south)
		(open patio kitchen north)
		(visited kitchen)
		(visited patio)
		(visited backyard)
	)
	(:goal
		(exists (?loc - location) (and (not (visited ?loc)) (at ?loc)))
	)
)