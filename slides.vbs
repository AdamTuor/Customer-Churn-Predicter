Imports Microsoft.Office.Interop

Module Module1

    Sub Main()
        ' Create PowerPoint application object
        Dim pptApp As New PowerPoint.Application()

        ' Create a new presentation
        Dim pptPresentation As PowerPoint.Presentation = pptApp.Presentations.Add()

        ' Add slides
        AddTitleSlide(pptPresentation)
        AddIntroductionSlide(pptPresentation)
        AddDatasetOverviewSlide(pptPresentation)
        AddChurnAnalysisPhoneSlide(pptPresentation)
        AddMonthlyChargesPhoneSlide(pptPresentation)
        AddFeatureImportanceSlide(pptPresentation)
        AddPredictiveModelSlide(pptPresentation)
        AddChurnPreventionStrategiesSlide(pptPresentation)
        AddConclusionSlide(pptPresentation)
        AddQuestionsSlide(pptPresentation)

        ' Save the presentation
        pptPresentation.SaveAs("C:/Users/agp07/DATA_ANALYTICS/Group_Projects/project-4/Presentation.pptx")

        ' Close the presentation and PowerPoint application
        pptPresentation.Close()
        pptApp.Quit()
    End Sub

    Sub AddTitleSlide(ByVal presentation As PowerPoint.Presentation)
        Dim slide As PowerPoint.Slide = presentation.Slides.Add(1, PowerPoint.PpSlideLayout.ppLayoutTitle)
        slide.Shapes.Title.TextFrame.TextRange.Text = "Telecommunication Churn Analysis"
    End Sub

    Sub AddIntroductionSlide(ByVal presentation As PowerPoint.Presentation)
        Dim slide As PowerPoint.Slide = presentation.Slides.Add(2, PowerPoint.PpSlideLayout.ppLayoutText)
        slide.Shapes.Title.TextFrame.TextRange.Text = "Introduction"
        slide.Shapes.Placeholders(2).TextFrame.TextRange.Text = "Brief introduction about the project and its objectives"
    End Sub

    Sub AddDatasetOverviewSlide(ByVal presentation As PowerPoint.Presentation)
        Dim slide As PowerPoint.Slide = presentation.Slides.Add(3, PowerPoint.PpSlideLayout.ppLayoutText)
        slide.Shapes.Title.TextFrame.TextRange.Text = "Dataset Overview"
        slide.Shapes.Placeholders(2).TextFrame.TextRange.Text = "Mention the name of the dataset and its source"
        slide.Shapes.Placeholders(3).TextFrame.TextRange.Text = "Highlight the total number of customers in the dataset"
        slide.Shapes.Placeholders(4).TextFrame.TextRange.Text = "Show the percentage of churned customers (26.54%)"
        slide.Shapes.Placeholders(5).TextFrame.TextRange.Text = "Highlight the importance of analyzing churn"
    End Sub

    Sub AddChurnAnalysisPhoneSlide(ByVal presentation As PowerPoint.Presentation)
        Dim slide As PowerPoint.Slide = presentation.Slides.Add(4, PowerPoint.PpSlideLayout.ppLayoutText)
        slide.Shapes.Title.TextFrame.TextRange.Text = "Churn Analysis - Phone Services"
        slide.Shapes.Placeholders(2).TextFrame.TextRange.Text = "Mention the number of customers with phone services (6361)"
        slide.Shapes.Placeholders(3).TextFrame.TextRange.Text = "Show the percentage of churned customers among phone service users"
        slide.Shapes.Placeholders(4).TextFrame.TextRange.Text = "Provide insights or observations about the churn rate for phone service users"
    End Sub

    Sub AddMonthlyChargesPhoneSlide(ByVal presentation As PowerPoint.Presentation)
        Dim slide As PowerPoint.Slide = presentation.Slides.Add(5, PowerPoint.PpSlideLayout.ppLayoutText)
        slide.Shapes.Title.TextFrame.TextRange.Text = "Monthly Charges - Phone Services"
        slide.Shapes.Placeholders(2).TextFrame.TextRange.Text = "Mention the average monthly charges for customers with phone services ($64.76)"
        slide.Shapes.Placeholders(3).TextFrame.TextRange.Text
