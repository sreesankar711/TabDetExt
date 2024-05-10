import "./carousel.css"

import TabInput from "../tabs/tabInput/TabInput"
import TableDetect from "../tabs/tableDetect/TableDetect"
import UploadingFirst from "../tabs/uploadingFirst/UploadingFirst";
import { useEffect, useState } from "react";
import { motion } from "framer-motion"
import {  useSelector } from "react-redux";

import { useRef } from "react";

export default function Carousel() {

    const selected = useSelector(state => state.imageSelected.selected)
    const is_tab_detected = useSelector(state => state.tabDetect.is_tab_detected)

    const [activeIndex, setIndex] = useState(0);


    useEffect(() => {
        if (selected && is_tab_detected === false) {
            console.log('reached carousel')
            setIndex(1)
        }

    }, [selected,is_tab_detected])

    useEffect(() => {
        if (is_tab_detected) {
            setIndex(2)
        }
    }, [is_tab_detected])
    const updateIndex = (newIndex) => {
        if (newIndex < 0) {
            newIndex = 0;
        }
        else if (newIndex >= 3) {
            newIndex = 2
        }
        setIndex(newIndex)
    }



    const scrollRef = useRef(null)

    return (
        <>



            <div className="top-container-carousel" ref={scrollRef} >

                <motion.div
                    initial={{
                        opacity: 0,
                        y: 200
                    }}
                    whileInView={{
                        opacity: 1,
                        y: 0,
                    }
                    }
                    transition={{
                        duration: 1,
                    }}

                    className="motion-div1"
                >
                    <div id="text" className= {selected?"hidden" :  ""}>
                        <h1>Transforming raw data <br />into actionable insights <br />has never been easier.</h1>
                    </div>

                </motion.div>

                <motion.div
                    initial={{
                        opacity: 0,
                        y: 200
                    }}
                    whileInView={{
                        opacity: 1,
                        y: 0
                    }}
                    transition={{
                        duration: 1,
                       
                    }}
                >
                    <div className= {selected? "container-carousel2":"container-carousel"}>
                        <div className="inner-carousel" style={{ transform: `translate(-${activeIndex * 100}%)` }}>
                            
                            <TabInput />
                            <UploadingFirst index={activeIndex} />
                            <TableDetect updates= {updateIndex}/>
                        </div>

                    </div>
                </motion.div>


            </div>

            {/* <button onClick={() => { updateIndex(activeIndex + 1) }} className="btn-carousel">
                go leftd
            </button>

            <button onClick={() => { updateIndex(activeIndex - 1) }}>
                go right
            </button> */}
        </>
    )
}