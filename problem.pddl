(define (problem explore-house)
    (:domain house-exploration)
    (:objects
        kitchen - location
        patio - location
        backyard - location
        driveway - location
        street - location
        south - direction
        north - direction
        east - direction
        west - direction
    )
    (:init
        (at backyard)
        (door-open kitchen backyard south)
        (door-closed patio backyard west)
        (connected kitchen backyard south)
        (connected backyard kitchen north)
        (connected backyard driveway south)
        (connected backyard street east)
    )
    (:goal 
        (at driveway)
    )
)