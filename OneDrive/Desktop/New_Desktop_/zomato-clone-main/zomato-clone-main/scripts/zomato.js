import
 { 
    bodyInfo, 
    bodyPopularPlaces,
    bodyLocalities, 
    bodyAddExtraLocalities,
    bodyCuisinesNearMe, 
    bodyRestypesNearMe,
    bodyResChains,
    bodyCities,
    countries
 } 
 
 from "../data/utils/body.js";

let bodyInfoHTML = ` `;

bodyInfo.forEach((bodyInfoItem) => {


    bodyInfoHTML += 

    `   <a href="#" class = "explore-link">
            <div style = "
                background-image: url('data/images/${bodyInfoItem.name}');
                ">
            </div>
        </a>
    `
});

document.querySelector(".js-body-info").innerHTML = bodyInfoHTML;

let bodyPopularPlacesHTML = ` `;

bodyPopularPlaces.forEach((bodyPopularPlaceItem) => {

    bodyPopularPlacesHTML += 
    `
        <a href="#">
            <div style = "background-image: url('data/images/${bodyPopularPlaceItem.source}');">
                <div>
                    <p class = "info-text">${bodyPopularPlaceItem.count} ${bodyPopularPlaceItem.info}</p>
                    <p class = "no-of-places">${bodyPopularPlaceItem.count} Places</p>
                </div>
            </div>

        </a>

    `                
});

document.querySelector(".popular-places").innerHTML = bodyPopularPlacesHTML;

let bodyLocalitiesHTML = ``;


let bodyAddExtraLocalitiesHTML;

function generateLocalityHTML (bodyLocalityItem) {

    bodyLocalitiesHTML += 
        `
            <div>
                <div>
                    <span class = "place-name">
                        ${bodyLocalityItem.name}
                    </span>
                    <span class = "places-count">
                        ${bodyLocalityItem.count} places
                    </span>
                </div>
                <i class="fa fa-angle-right"></i>
            </div>
        `
}

function appendSeeMoreContainer () {

    bodyLocalitiesHTML +=

        ` <div class = "see-more-text js-see-more-text">
                <div>see more</div>
                <i class="fa fa-angle-down"></i>
                                    
            </div>
        ` ;
}

function appendSeeLessContainer () {
    bodyLocalitiesHTML +=

        ` <div class = "see-less-text js-see-less-text">
                    <div>see less</div>
                    <i class="fa fa-angle-up"></i>               
                </div>
        `

}

bodyLocalities.forEach((bodyLocalityItem) => {

    generateLocalityHTML(bodyLocalityItem);

});

bodyAddExtraLocalitiesHTML = bodyLocalitiesHTML;

appendSeeMoreContainer();

document.querySelector(".places-list").innerHTML = bodyLocalitiesHTML;

document.querySelector(".js-see-more-text").
        addEventListener("click", (event) => {

            const target = event.target;

            bodyLocalitiesHTML = bodyAddExtraLocalitiesHTML;

            bodyAddExtraLocalities.forEach((bodyLocalityItem) => {
                generateLocalityHTML(bodyLocalityItem);
            });

            appendSeeLessContainer ();

            document.querySelector(".places-list").innerHTML = bodyLocalitiesHTML;
});

let count1 = 0;

function fillCuisinesHTML (bodyCuisinesNearMeHTML) {

    bodyCuisinesNearMe.forEach((bodyCuisineItem) => {
        bodyCuisinesNearMeHTML += 
            `
            ${bodyCuisineItem} food near me &nbsp <span style = "font-weight: bold;">-</span> &nbsp
            `
    });

    bodyCuisinesNearMeHTML += `</div>`;

    return bodyCuisinesNearMeHTML;
}

document.querySelector(".js-click-cuisines").
    addEventListener("click", () => {

        let bodyCuisinesNearMeHTML = `<div style = "padding-top: 0.5rem;">`;

        if(count1 === 0) {
            bodyCuisinesNearMeHTML = fillCuisinesHTML(bodyCuisinesNearMeHTML)

            document.querySelector(".js-cuisines-near-me").
                innerHTML = bodyCuisinesNearMeHTML;
                count1 += 1;
        }
        else {
            bodyCuisinesNearMeHTML = ` `;
            document.querySelector(".js-cuisines-near-me").
                    innerHTML = bodyCuisinesNearMeHTML;
            count1 -= 1;
        }

        
    });

let count2 = 0;

function fillRestypesHTML (bodyRestypesNearMeHTML) {

    bodyRestypesNearMe.forEach((bodyRestypesItem) => {

        bodyRestypesNearMeHTML += 
            `
            ${bodyRestypesItem} near me &nbsp <span style = "font-weight: bold;">-</span> &nbsp
            `
    });

    bodyRestypesNearMeHTML += `</div>`;

    return bodyRestypesNearMeHTML;
}

document.querySelector(".js-click-restypes").
    addEventListener("click", () => {

        let bodyRestypesNearMeHTML = `<div style = "padding-top: 0.5rem;">`;

        if(count2 === 0) {
            bodyRestypesNearMeHTML = fillRestypesHTML(bodyRestypesNearMeHTML);

            document.querySelector(".js-restypes-near-me").
                innerHTML = bodyRestypesNearMeHTML;

                count2 += 1;
        }

        else {
            bodyRestypesNearMeHTML = ` `;
            document.querySelector(".js-restypes-near-me").
                    innerHTML = bodyRestypesNearMeHTML;
            count2 -= 1;
        }

    })

let count3 = 0;

function fillResChainsHTML (bodyResChainsHTML) {
    bodyResChains.forEach((bodyResChainItem) => {
       
        bodyResChainsHTML += 

        ` 
            <div>${bodyResChainItem}</div>
        `
    });

    bodyResChainsHTML +=  `</div>`;

    return bodyResChainsHTML ;
}

document.querySelector(".js-click-res-chains").
    addEventListener("click", () => {

        let bodyResChainsHTML = `<div style = 
        "  padding-top: 1rem;
           display: grid;
           grid-template-columns: repeat(5, 1fr);
           align-items: center;
        ">`;

        if(count3 === 0) {

            bodyResChainsHTML = fillResChainsHTML(bodyResChainsHTML);

             document.querySelector(".js-top-res-chains").
                innerHTML = bodyResChainsHTML;

                count3 += 1;
        }

        else {
            bodyResChainsHTML = ` `;

            document.querySelector(".js-top-res-chains").
                    innerHTML = bodyResChainsHTML;
            count3 -= 1;
        }

    })
let count4 = 0;
    
function fillCitiesHTML (bodyCitiesHTML) {

    bodyCities.forEach((bodyCityItem) => {

        bodyCitiesHTML += 

        ` <div style = "padding: 0.5rem 0rem;">${bodyCityItem}</div>
        `     
    });

    bodyCitiesHTML += `</div>`

    return bodyCitiesHTML;
}

document.querySelector(".js-click-cities").
    addEventListener("click", () => {

        let bodyCitiesHTML = `<div style =
         "  padding-top: 0.5rem;
            display: grid;
            grid-template-columns: repeat(5,1fr);
         ">`;

        if(count4 === 0) {

            bodyCitiesHTML = fillCitiesHTML(bodyCitiesHTML);

            document.querySelector(".js-we-deliver-to").
                innerHTML = bodyCitiesHTML;

                count4 += 1;
        }

        else {
            bodyCitiesHTML = ``;

            document.querySelector(".js-we-deliver-to").
                innerHTML = bodyCitiesHTML;

                count4 -= 1;
        }
    })



let countriesHTML = ` `;

function generateCountriesHTML () {
   countries.forEach((country) => {
        countriesHTML += 
            `   <div style = 
                    "   font-size: 0.8rem;
                        display: flex;
                        color: gray;
                        gap: 1rem;
                        align-items: center;
                    ">
                    <img style = " height: 1rem;"
                    src = "${country.flag}">
                    <div>${country.name}</div>
                </div>
            `
    })
}



generateCountriesHTML ();

document.querySelector(".js-country-absolute"). 
innerHTML = countriesHTML;

             