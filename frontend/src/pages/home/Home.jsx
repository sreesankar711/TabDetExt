import "./home.css"
import { useEffect, useRef, useState } from "react";
import Typed from 'typed.js'
import { IoArrowDownCircleOutline } from "react-icons/io5";

function ScrollIcon() {
    const [isScrolled, setIsScrolled] = useState(false);
    
    useEffect(() => {
        const handleScroll = () => {
            // Check if the user has scrolled past a certain threshold (e.g., 100 pixels)
            if (window.scrollY > window.innerHeight - 400) {
                setIsScrolled(true); 
            } else {
                setIsScrolled(false); // Set state to false if scrolled back up
            }
        };
       
        // Add scroll event listener when component mounts
        window.addEventListener('scroll', handleScroll);

        // Clean up by removing scroll event listener when component unmounts
        return () => {
            window.removeEventListener('scroll', handleScroll);

        };
    }, []); // Empty dependency array ensures the effect runs only once when component mounts

    const scrollToNextPage = () => {
        // Scroll to the next page
        window.scrollTo({ top: window.innerHeight, behavior: 'smooth' });
    };

    return (

        <div id="icon-final" className={isScrolled ? 'hidden' : '' } onClick={scrollToNextPage}>
            {<IoArrowDownCircleOutline className="scroll-icon-home" />}
        </div>

    );
}


function Autotype() {
    const el = useRef("");

    useEffect(() => {
        const typed = new Typed(el.current, {
            strings: ['Data Extraction. ^1000',
                ' Structure Detection. ^1000'],
            typeSpeed: 50,
            backSpeed: 40,

            loop: true
        });

        return () => {
            // Destroy Typed instance during cleanup to stop animation
            typed.destroy();
        };
    }, []);
    return (
        <div>
            <div className="auto">
                <span ref={el}></span>
            </div>
        </div>
    )
}

export default function Home() {
    return (
        <>
            <div className="container-home" style={{ backgroundImage: "url(/wallpaper1.jpg)" }}>
                <div className="nav-home">
                    <div className="nav-left">
                        <h1>Tabular Data Extraction</h1>
                    </div>
                    <div className="nav-right">
                        <ul>
                            <li><a href="#">Home</a></li>
                            <li><a href="#">About Us</a></li>
                            <li><a href="#">Contact Us</a></li>

                        </ul>
                    </div>
                </div>

                <div className="hero-home">
                    <span>Tabular</span>
                    <Autotype />
                </div>

                <div className="icon-container-home">
                    <ScrollIcon className="downward-icon" />
                </div>
            </div>




        </>
    )
}