jQuery(document).ready(function($) {

    // Sticky Nav Bar
    $(window).scroll(function() {
        if ($(this).scrollTop() > 20) {
            $('.sticky').addClass("fixed");
        }
        else {
            $('.sticky').removeClass("fixed");
        }
    });

    // Get current year for copyright description
    date = new Date();
    thisYear = date.getFullYear();
    document.getElementById("thisYear").innerHTML = thisYear;

});
